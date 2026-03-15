import torch
import numpy as np
from typing import Optional, Union, List
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich import box
from model.tokenizer import Tokenizer
from pipeline.component_index import component_idx as build_component_idx, split_component_idx
from store.logit_context import logit_ctx
from store.top_coactivation import top_coactivation
from store.context import mid_ctx, neg_ctx
from sae.topk_sae import SAEConfig


class Display:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.console = Console()

    # ── Token decoding ──────────────────────────────────────────────

    def _resolve_token_parts(self, tokens):
        """Decode a token list into display-ready text parts that preserve spacing."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        elif isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        full_decoded = self.tokenizer.decode(tokens)
        remaining = full_decoded
        parts = []

        for token_id in tokens:
            decoded_token = self.tokenizer.decode([token_id])

            if remaining.startswith(decoded_token):
                parts.append(decoded_token)
                remaining = remaining[len(decoded_token):]
            else:
                found = False
                for j in range(1, len(remaining) + 1):
                    if remaining[:j].strip() == decoded_token.strip():
                        parts.append(remaining[:j])
                        remaining = remaining[j:]
                        found = True
                        break
                if not found:
                    parts.append(decoded_token)

        return tokens, parts

    # ── Intensity / color mapping ───────────────────────────────────

    def _compute_intensities(self, values):
        """Z-score normalize activation values to [0, 1] intensities (-1 = no activation)."""
        if isinstance(values, torch.Tensor):
            values_list = values.tolist()
        elif isinstance(values, np.ndarray):
            values_list = values.tolist()
        else:
            values_list = list(values)

        non_zero = np.array([v for v in values_list if v > 0])
        v_mean = float(np.mean(non_zero)) if non_zero.size > 0 else 0.0
        v_std  = float(np.std(non_zero))  if non_zero.size > 1 else 1.0

        intensities = []
        for val in values_list:
            if val <= 0:
                intensities.append(-1.0)
            else:
                z = (val - v_mean) / (v_std + 1e-6)
                intensities.append(float(np.clip((z + 1) / 4, 0, 1)))
        return intensities

    @staticmethod
    def _intensity_to_style(intensity: float, scheme: str = "top") -> str:
        """
        Map a [0, 1] intensity to a rich style string.

        scheme="top"  — warm amber → deep red  (high-activation sequences)
        scheme="mid"  — teal → deep cyan        (moderate-activation sequences)
        """
        if intensity < 0:
            return ""
        if intensity < 0.15:
            return "dim"

        if scheme == "mid":
            r = int(20  * (1 - intensity))
            g = min(255, 140 + int(80 * intensity))
            b = min(255, 160 + int(60 * intensity))
            fg = "white" if intensity > 0.35 else "#1a1a1a"
        else:
            r = min(255, 180 + int(75 * intensity))
            g = int(140 * (1 - intensity))
            b = int(50  * (1 - intensity))
            fg = "white" if intensity > 0.45 else "#1a1a1a"

        return f"{fg} on rgb({r},{g},{b})"

    # ── Building styled text ────────────────────────────────────────

    def build_sequence_text(
        self,
        tokens: Union[torch.Tensor, np.ndarray, List[int]],
        values: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
        scheme: str = "top",
    ) -> Text:
        """Build a rich Text object with optional heatmap coloring."""
        _, parts = self._resolve_token_parts(tokens)

        intensities = self._compute_intensities(values) if values is not None else None

        text = Text()
        for i, part in enumerate(parts):
            if intensities is not None:
                style = self._intensity_to_style(intensities[i], scheme=scheme)
                text.append(part, style=style)
            else:
                text.append(part)
        return text

    # ── Printing ────────────────────────────────────────────────────

    def print_sequence(
        self,
        tokens: Union[torch.Tensor, np.ndarray, List[int]],
        values: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
        title: Optional[str] = None,
        border_style: str = "bright_black",
        scheme: str = "top",
    ):
        """Print a token sequence inside a rounded panel with optional heatmap."""
        text = self.build_sequence_text(tokens, values, scheme=scheme)
        panel = Panel(
            text,
            title=title,
            title_align="left",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(0, 1),
        )
        self.console.print(panel)

    # ── High-level analysis helpers ─────────────────────────────────

    def analyze_and_print_top_latents(self, top_ctx, model, bank, loader, n_latents=3, n_sequences=5):
        """Identify top latents (normalized per component) and display their analysis."""
        self.console.rule(
            f"[bold]Top {n_latents} latents by highest activation score[/bold]  "
            "[dim](normalized per component)[/dim]"
        )

        raw_max_vals = top_ctx.ctx_seq_val[:, :, 0].float()
        component_maxes = raw_max_vals.max(dim=1, keepdim=True).values
        max_vals = raw_max_vals / component_maxes.clamp(min=1e-6)

        topk_vals, topk_flat_indices = torch.topk(max_vals.flatten(), k=n_latents)

        latents_info = []
        for i in range(n_latents):
            flat_idx = int(topk_flat_indices[i].item())
            norm_val = topk_vals[i].item()

            component_idx = int(flat_idx // bank.d_sae)
            latent_idx    = int(flat_idx %  bank.d_sae)
            raw_val = raw_max_vals[component_idx, latent_idx].item()

            seq_ids  = top_ctx.ctx_seq_idx[component_idx, latent_idx, :n_sequences].tolist()
            seq_vals = top_ctx.ctx_seq_val[component_idx, latent_idx, :n_sequences].tolist()
            valid_seqs = [(sid, sv) for sid, sv in zip(seq_ids, seq_vals) if sv > 0]

            latents_info.append({
                "component_idx": component_idx,
                "latent_idx":    latent_idx,
                "norm_val":      norm_val,
                "raw_val":       raw_val,
                "sequences":     valid_seqs,
                **_build_mid_neg_seqs(component_idx, latent_idx, n_sequences),
            })

        self.analyze_and_print_latents(model, bank, loader, latents_info)

    def analyze_and_print_specific_latent(self, top_ctx, model, bank, loader, layer_idx, kind, latent_idx, n_sequences=5):
        """Analyze and display a specific latent by layer, kind, and index."""
        if kind not in bank.kinds:
            raise ValueError(f"Invalid kind: {kind}. Expected one of {bank.kinds}")

        component_idx_value = build_component_idx(layer_idx, bank.kinds.index(kind), len(bank.kinds))

        raw_max_vals = top_ctx.ctx_seq_val[component_idx_value, :, 0].float()
        component_max = raw_max_vals.max().item()
        raw_val  = top_ctx.ctx_seq_val[component_idx_value, latent_idx, 0].item()
        norm_val = raw_val / (component_max if component_max > 1e-6 else 1.0)

        seq_ids  = top_ctx.ctx_seq_idx[component_idx_value, latent_idx, :n_sequences].tolist()
        seq_vals = top_ctx.ctx_seq_val[component_idx_value, latent_idx, :n_sequences].tolist()
        valid_seqs = [(sid, sv) for sid, sv in zip(seq_ids, seq_vals) if sv > 0]

        latents_info = [{
            "component_idx": component_idx_value,
            "latent_idx":    latent_idx,
            "norm_val":      norm_val,
            "raw_val":       raw_val,
            "sequences":     valid_seqs,
            **_build_mid_neg_seqs(component_idx_value, latent_idx, n_sequences),
        }]

        self.analyze_and_print_latents(model, bank, loader, latents_info)

    def analyze_and_print_latents(self, model, bank, loader, latents_info):
        """Re-run the model and display per-token activations for each latent."""
        # Collect sequences that need a model rerun (top + mid both have activation values).
        needed_sequences: set = set()
        for info in latents_info:
            needed_sequences.update(s[0] for s in info["sequences"])
            needed_sequences.update(s[0] for s in info.get("mid_sequences", []))

        # Re-run pass for token-level activations (top + mid).
        token_activations_map = {}
        if needed_sequences:
            self.console.print(
                f"\n[dim]Re-running model for {len(needed_sequences)} sequences "
                f"to get token-level activations...[/dim]"
            )

            def rerun_sae_callback(layer_idx, batch_ids, activations):
                n_kinds = len(bank.kinds)
                for kind_idx, kind in enumerate(bank.kinds):
                    comp_idx = build_component_idx(layer_idx, kind_idx, n_kinds)
                    relevant = [info for info in latents_info if info["component_idx"] == comp_idx]
                    if not relevant:
                        continue
                    with torch.no_grad():
                        latents = bank.encode(activations[kind_idx], kind, layer_idx)
                        top_acts    = latents[0]
                        top_indices = latents[1]
                        for info in relevant:
                            l_idx = info["latent_idx"]
                            for b_idx, seq_id in enumerate(batch_ids.tolist()):
                                mask = (top_indices[b_idx] == l_idx)
                                seq_token_acts = (top_acts[b_idx] * mask).sum(dim=-1)
                                token_activations_map[(seq_id, comp_idx, l_idx)] = seq_token_acts

            for batch_ids, batch_tokens in loader.get_batches_by_ids(sorted(needed_sequences)):
                assert isinstance(batch_tokens, torch.Tensor)
                model.forward(
                    batch_tokens,
                    num_gen=1,
                    tokenize_final=False,
                    activations_callback=lambda l, a: rerun_sae_callback(l, batch_ids, a),
                    return_activations=False
                )

        # Display results.
        for info in latents_info:
            component_idx = info["component_idx"]
            latent_idx    = info["latent_idx"]
            layer_idx, kind_idx = split_component_idx(component_idx, len(bank.kinds))
            kind = bank.kinds[kind_idx]

            self.console.print()
            self.console.rule(
                f"[bold cyan]Latent {latent_idx}[/bold cyan]  "
                f"[dim]|[/dim]  {kind} layer {layer_idx}  "
                f"[dim]|[/dim]  norm [bold]{info['norm_val']:.4f}[/bold]  "
                f"raw [bold]{info['raw_val']:.4f}[/bold]"
            )

            # ── Top Context ─────────────────────────────────────────
            if info["sequences"]:
                self.console.print(
                    "[bold bright_white]Top Context[/bold bright_white] "
                    "[dim]highest activation sequences[/dim]"
                )
            for seq_id, seq_val in info["sequences"]:
                tokens = loader.get_sequence(seq_id)
                per_token_vals = token_activations_map.get((seq_id, component_idx, latent_idx))
                self.print_sequence(
                    tokens,
                    per_token_vals,
                    title=f"[bold]Seq {seq_id}[/bold] [dim]score {seq_val:.4f}[/dim]",
                    border_style="bright_black",
                    scheme="top",
                )

            # ── Mid Context ─────────────────────────────────────────
            mid_seqs = info.get("mid_sequences", [])
            if mid_seqs:
                self.console.print(
                    "\n[bold cyan]Mid Context[/bold cyan] "
                    "[dim]reservoir-sampled moderate-activation sequences[/dim]"
                )
                for seq_id, seq_val in mid_seqs:
                    tokens = loader.get_sequence(seq_id)
                    per_token_vals = token_activations_map.get((seq_id, component_idx, latent_idx))
                    self.print_sequence(
                        tokens,
                        per_token_vals,
                        title=f"[bold]Seq {seq_id}[/bold] [dim]score {seq_val:.4f}[/dim]",
                        border_style="cyan",
                        scheme="mid",
                    )

            # ── Neg Context ─────────────────────────────────────────
            neg_seqs = info.get("neg_sequences", [])
            if neg_seqs:
                self.console.print(
                    "\n[bold blue]Neg Context[/bold blue] "
                    "[dim]similar sequences that do not activate this latent[/dim]"
                )
                for seq_id, cos_sim in neg_seqs:
                    tokens = loader.get_sequence(seq_id)
                    self.print_sequence(
                        tokens,
                        values=None,
                        title=f"[bold]Seq {seq_id}[/bold] [dim]cosine sim {cos_sim:.4f}[/dim]",
                        border_style="blue",
                        scheme="top",
                    )

            # ── Logit Context ───────────────────────────────────────
            top_tokens = logit_ctx.get_top_tokens(component_idx, latent_idx)
            if top_tokens:
                self.console.print()
                token_texts = []
                for t_id, prob in top_tokens[:12]:
                    t_str = self.tokenizer.decode([t_id])
                    t_str = t_str.replace("[", "\\[")
                    token_texts.append(f"[bold green]{t_str!r}[/bold green] [dim]{prob:.4f}[/dim]")

                self.console.print(Panel(
                    ", ".join(token_texts),
                    title="[bold green]Logit Context[/bold green] [dim](predicted tokens)[/dim]",
                    title_align="left",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(0, 1),
                ))

            # ── Top Co-activation ──────────────────────────────────
            self.console.print()
            co_ids  = top_coactivation.top_indices[component_idx, latent_idx]
            co_vals = top_coactivation.top_values[component_idx, latent_idx]
            d_sae   = SAEConfig().d_sae
            kind_names = bank.kinds

            co_parts = []
            for i in range(co_ids.shape[0]):
                gid = co_ids[i].item()
                val = co_vals[i].item()
                if gid == 0 and val == 0:
                    continue
                co_comp  = gid // d_sae
                co_lat   = gid % d_sae
                co_layer, co_kind_idx = split_component_idx(co_comp, len(kind_names))
                co_kind = kind_names[co_kind_idx]
                co_parts.append(
                    f"[bold magenta]{co_layer} {co_kind} {co_lat}[/bold magenta] "
                    f"[dim]{val:.3f}[/dim]"
                )

            if co_parts:
                self.console.print(Panel(
                    ", ".join(co_parts),
                    title="[bold magenta]Co-Magnitude[/bold magenta] [dim](top co-occurring latents)[/dim]",
                    title_align="left",
                    border_style="magenta",
                    box=box.ROUNDED,
                    padding=(0, 1),
                ))

        self.console.print()


# ---------------------------------------------------------------------------
# Module-level helper — build mid/neg sequence lists for a single latent.
# Returns an empty dict if either store has no data yet (before load()).
# ---------------------------------------------------------------------------

def _build_mid_neg_seqs(
    component_idx: int,
    latent_idx: int,
    n_sequences: int,
) -> dict:
    mid_ids  = mid_ctx.ctx_seq_idx[component_idx, latent_idx, :n_sequences].tolist()
    mid_vals = mid_ctx.ctx_seq_val[component_idx, latent_idx, :n_sequences].tolist()
    valid_mid = [(int(sid), float(sv)) for sid, sv in zip(mid_ids, mid_vals) if sv > 0]

    neg_ids  = neg_ctx.ctx_seq_idx[component_idx, latent_idx, :n_sequences].tolist()
    neg_vals = neg_ctx.ctx_seq_val[component_idx, latent_idx, :n_sequences].tolist()
    valid_neg = [(int(sid), float(sv)) for sid, sv in zip(neg_ids, neg_vals) if sv > 0]

    return {"mid_sequences": valid_mid, "neg_sequences": valid_neg}


display = Display()
