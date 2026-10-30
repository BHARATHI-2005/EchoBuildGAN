from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from ecobuild.inference import (
    CheckpointError,
    InvalidImageError,
    checkpoint_path,
    decode_image,
    generate_image,
    load_generator,
)
from ecobuild.image_provider import (
    DEFAULT_CF_MODEL,
    ImageProviderConfigurationError,
    ImageProviderRequestError,
    generate_cloudflare_image,
    validate_cloudflare_account_id,
)
from ecobuild.reports import build_design_report


ROOT = Path(__file__).resolve().parent
PREFERENCES = [
    "Rooftop greenery",
    "Landscaping and vegetation",
    "Natural-lighting concepts",
    "Solar-panel concepts",
    "Sustainable facade concepts",
    "Green courtyards",
]

load_dotenv(ROOT / ".env")


def _config_value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        return str(st.secrets.get(name, default))
    except StreamlitSecretNotFoundError:
        return default


st.set_page_config(
    page_title="EcoBuild GAN | Sustainable Architecture Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _apply_theme() -> None:
    background_path = ROOT / "imgs" / "ecobuild-bg.png"
    background = ""
    if background_path.is_file():
        encoded = base64.b64encode(background_path.read_bytes()).decode("ascii")
        background = (
            f'background-image: linear-gradient(rgba(247, 246, 232, .83), '
            f'rgba(247, 246, 232, .90)), url("data:image/png;base64,{encoded}");'
        )
    st.markdown(
        f"""
        <style>
        :root {{
          --forest: #244b3b; --sage: #6f9275; --cream: #f7f6e8;
          --olive: #dce5ca; --ink: #20342b; --text-color: #20342b;
          --background-color: #f7f6e8; --secondary-background-color: #e8eedb;
          --primary-color: #426d51;
        }}
        .stApp {{
          {background}
          background-color: var(--cream);
          background-size: cover; background-attachment: fixed;
          color: var(--ink); color-scheme: light;
        }}
        [data-testid="stHeader"] {{ background: rgba(247,246,232,.55); }}
        [data-testid="stSidebar"] {{
          background: rgba(232, 238, 219, .84);
          border-right: 1px solid rgba(36,75,59,.13);
        }}
        [data-testid="stSidebar"] * {{ color: var(--ink) !important; }}
        [data-testid="stMain"] h1, [data-testid="stMain"] h2,
        [data-testid="stMain"] h3, [data-testid="stMain"] p,
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
        [data-testid="stCaptionContainer"], .stApp label {{
          color: var(--ink);
        }}
        [data-testid="stMain"] {{ animation: fadein .45s ease-out; }}
        @keyframes fadein {{ from {{ opacity: .4; transform: translateY(5px); }}
          to {{ opacity: 1; transform: translateY(0); }} }}
        .hero {{
          padding: clamp(1.5rem, 5vw, 4rem); border-radius: 28px;
          background: linear-gradient(120deg, rgba(28,66,50,.94),
             rgba(77,112,79,.88));
          color: #fffdf2; box-shadow: 0 18px 48px rgba(32,52,43,.18);
          margin: .5rem 0 1.5rem;
        }}
        .hero h1 {{ color: #fffdf2 !important; font-size: clamp(2.3rem, 5vw, 4rem); }}
        .hero p {{ color: #edf2e4 !important; font-size: 1.08rem; max-width: 760px; }}
        .eyebrow {{ text-transform: uppercase; letter-spacing: .16em;
          font-size: .78rem; font-weight: 700; color: #dce8c8 !important; }}
        .glass-card {{
          background: rgba(255,255,248,.82); border: 1px solid rgba(36,75,59,.13);
          border-radius: 20px; padding: 1.2rem 1.3rem; height: 100%;
          box-shadow: 0 10px 28px rgba(32,52,43,.08);
          transition: transform .18s ease, box-shadow .18s ease;
        }}
        .glass-card:hover {{ transform: translateY(-3px);
          box-shadow: 0 14px 34px rgba(32,52,43,.13); }}
        .glass-card h3 {{ color: var(--forest); margin-top: .2rem; }}
        .muted {{ color: #52665a; }}
        .stApp [data-baseweb="input"] > div,
        .stApp [data-baseweb="textarea"] > div,
        .stApp [data-baseweb="select"] > div {{
          background-color: rgba(255,255,255,.94);
          border-color: rgba(36,75,59,.28);
        }}
        .stApp [data-baseweb="input"] input,
        .stApp [data-baseweb="textarea"] textarea,
        .stApp [data-baseweb="select"] * {{ color: var(--ink) !important; }}
        .stApp input::placeholder, .stApp textarea::placeholder {{
          color: #64756a !important;
          opacity: 1 !important;
        }}
        [data-testid="stFileUploader"] section {{
          background: rgba(255,255,255,.82);
          border-color: rgba(36,75,59,.3);
        }}
        [data-testid="stFileUploader"] * {{ color: var(--ink); }}
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button * {{
          background: #fffdf2 !important; color: var(--ink) !important;
          border-color: rgba(36,75,59,.28) !important;
        }}
        [data-testid="stCode"] {{
          max-width: 100%;
          min-width: 0;
        }}
        [data-testid="stCode"] pre,
        [data-testid="stCode"] code {{
          white-space: pre-wrap !important;
          overflow-wrap: anywhere !important;
          word-break: break-word;
          max-width: 100%;
        }}
        .notice {{
          padding: .9rem 1rem; border-radius: 14px;
          background: rgba(220,229,202,.72);
          border-left: 4px solid #6f9275; margin: .8rem 0;
        }}
        .stButton > button, .stDownloadButton > button {{
          border-radius: 12px; transition: transform .15s ease;
          background: #315c47 !important; color: #fffdf2 !important;
          border-color: #315c47 !important;
        }}
        .stButton > button p, .stDownloadButton > button p {{
          color: #fffdf2 !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
          transform: translateY(-1px);
        }}
        @media (max-width: 700px) {{
          .hero {{ padding: 1.5rem; border-radius: 20px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _cached_generator(
    checkpoint: str,
    device_name: str,
    net_g: str,
    ngf: int,
    norm: str,
    use_dropout: bool,
):
    return load_generator(
        checkpoint,
        device_name,
        net_g=net_g,
        ngf=ngf,
        norm=norm,
        use_dropout=use_dropout,
    )


def _generator_settings() -> tuple[Path, str, int, str, bool]:
    checkpoints_dir = Path(
        os.getenv("ECOBUILD_CHECKPOINTS_DIR", "checkpoints")
    )
    if not checkpoints_dir.is_absolute():
        checkpoints_dir = ROOT / checkpoints_dir
    model_name = os.getenv("ECOBUILD_MODEL_NAME", "ecobuild_pix2pix")
    epoch = os.getenv("ECOBUILD_MODEL_EPOCH", "latest")
    return (
        checkpoint_path(checkpoints_dir, model_name, epoch),
        os.getenv("ECOBUILD_NET_G", "unet_256"),
        int(os.getenv("ECOBUILD_NGF", "64")),
        os.getenv("ECOBUILD_NORM", "batch"),
        os.getenv("ECOBUILD_USE_DROPOUT", "true").strip().lower()
        in {"1", "true", "yes"},
    )


def _display_checkpoint_path(checkpoint: Path) -> str:
    try:
        return checkpoint.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(checkpoint)


def _image_generation_prompt(
    description: str, preferences: list[str]
) -> str:
    preference_text = (
        ", ".join(preferences)
        if preferences
        else "green roof planting, integrated solar panels, native landscaping, "
        "daylight, and a sustainable facade"
    )
    project_context = description.strip() or "a sustainable residential building"
    return (
        "Transform this uploaded architectural sketch into a polished, "
        "photorealistic sustainable architecture visualization. Treat the "
        "sketch as the design reference: preserve its camera viewpoint, main "
        "building outline, floor count, window and door placement, and overall "
        "layout. Render realistic materials, natural daylight, believable "
        "landscaping, and refined architectural details. Incorporate these "
        f"requested concepts where they fit: {preference_text}. "
        f"Project context: {project_context}. Produce one coherent exterior "
        "architectural concept image with no text, labels, or watermark. This "
        "is a visual concept, not a construction or engineering plan."
    )


def _hero(title: str, tagline: str, body: str) -> None:
    st.markdown(
        f'<section class="hero"><div class="eyebrow">{tagline}</div>'
        f"<h1>{title}</h1><p>{body}</p></section>",
        unsafe_allow_html=True,
    )


def _navigate_to_studio() -> None:
    st.session_state["page"] = "Design Studio"


def _render_home() -> None:
    _hero(
        "EcoBuild GAN",
        "Transform Ideas into Sustainable Spaces",
        "Explore architectural concepts from your building sketches. "
        "Generated imagery is a creative starting point—not a verified plan.",
    )
    st.button(
        "Start designing  →",
        type="primary",
        key="start_designing",
        on_click=_navigate_to_studio,
    )

    st.subheader("A thoughtful concept workflow")
    steps = st.columns(3)
    for column, icon, title, copy in zip(
        steps,
        ["✎", "✦", "↗"],
        ["Bring a sketch", "Explore a concept", "Review with experts"],
        [
            "Upload a PNG or JPEG architectural image.",
            "Generate a hosted image-edit concept, or use local Pix2Pix weights.",
            "Discuss ideas with professionals before making design decisions.",
        ],
    ):
        with column:
            st.markdown(
                f'<div class="glass-card"><span>{icon}</span><h3>{title}</h3>'
                f'<p class="muted">{copy}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")
    backend = _config_value(
        "ECOBUILD_IMAGE_PROVIDER", "cloudflare"
    ).strip().lower()
    if backend == "cloudflare":
        if (
            _config_value("ECOBUILD_CF_ACCOUNT_ID").strip()
            and _config_value("ECOBUILD_CF_API_TOKEN").strip()
        ):
            st.success(
                "Cloudflare Workers AI image editing is configured. The Design Studio "
                "asks for consent before sending an uploaded image to the "
                "Cloudflare model."
            )
        else:
            st.info(
                "Add your Cloudflare account ID and Workers AI API token to "
                "your local `.env` to enable free-tier sketch-to-design "
                "generation."
            )
    elif backend == "pix2pix":
        st.info(
            "Local Pix2Pix mode is selected. Add a trained architectural "
            "generator checkpoint to enable image generation."
        )
    else:
        st.warning(
            "ECOBUILD_IMAGE_PROVIDER must be set to `cloudflare` or `pix2pix`."
        )


def _render_studio() -> None:
    _hero(
        "AI Design Studio",
        "Sketch → concept",
        "Upload an architectural image and explore a generated visual concept "
        "using Cloudflare Workers AI or a local Pix2Pix checkpoint.",
    )
    if st.button("Reset current design", key="reset_design"):
        for key in (
            "design_upload",
            "design_bytes",
            "design_image",
            "generated_image",
            "design_filename",
            "project_description",
            "design_preferences",
            "remote_image_consent",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    uploaded = st.file_uploader(
        "Choose a building sketch or architectural image",
        type=["png", "jpg", "jpeg"],
        key="design_upload",
        help="PNG and JPEG images are accepted. Images over 25 megapixels are rejected.",
    )
    original = None
    if uploaded is not None:
        content = uploaded.getvalue()
        if content != st.session_state.get("design_bytes"):
            st.session_state.pop("design_bytes", None)
            st.session_state["generated_image"] = None
            st.session_state["remote_image_consent"] = False
            st.session_state["design_filename"] = uploaded.name
        try:
            original = decode_image(content)
            st.session_state["design_bytes"] = content
            st.session_state["design_image"] = original
        except InvalidImageError as error:
            st.error(str(error))
            st.session_state["design_image"] = None
            st.session_state.pop("design_bytes", None)
    elif st.session_state.get("design_bytes"):
        try:
            original = decode_image(st.session_state["design_bytes"])
            st.session_state["design_image"] = original
        except InvalidImageError as error:
            st.error(str(error))
            st.session_state.pop("design_bytes", None)
            st.session_state["design_image"] = None
    else:
        st.session_state["design_image"] = None

    description = st.text_area(
        "Project description",
        placeholder="For example: a compact family home on a shaded urban lot...",
        key="project_description",
    )
    preferences = st.multiselect(
        "Sustainability concept preferences",
        PREFERENCES,
        key="design_preferences",
    )
    image_provider = _config_value(
        "ECOBUILD_IMAGE_PROVIDER", "cloudflare"
    ).strip().lower()
    if image_provider == "cloudflare":
        st.caption(
            "Your project description and preferences guide Cloudflare's "
            "image-edit model. The uploaded image and prompt are sent to "
            "Cloudflare only after you give consent."
        )
    else:
        st.caption(
            "The local Pix2Pix generator is not conditioned on these preference "
            "selections."
        )
    if original is not None:
        st.subheader("Your image")
        filename = (
            uploaded.name
            if uploaded is not None
            else st.session_state.get("design_filename", "Current session image")
        )
        st.image(original, caption=filename, width="stretch")
        if uploaded is None:
            st.caption(
                "The current image is retained in this session. Use Reset "
                "current design to clear it."
            )
        if image_provider == "cloudflare":
            account_id = _config_value("ECOBUILD_CF_ACCOUNT_ID").strip()
            token = _config_value("ECOBUILD_CF_API_TOKEN").strip()
            model_name = _config_value(
                "ECOBUILD_CF_MODEL", DEFAULT_CF_MODEL
            ).strip()
            account_error = None
            if account_id:
                try:
                    validate_cloudflare_account_id(account_id)
                except ImageProviderConfigurationError as error:
                    account_error = str(error)
            consent = st.checkbox(
                "I consent to sending this image and prompt to Cloudflare "
                "Workers AI for image generation.",
                key="remote_image_consent",
            )
            st.caption(
                "Cloudflare's free Workers plan currently includes a daily "
                "AI allowance. Requests beyond that allowance may require "
                "a paid plan. Cloudflare processes the uploaded image."
            )
            if not account_id or not token:
                st.warning(
                    "Cloudflare Workers AI is not configured. Set "
                    "ECOBUILD_CF_ACCOUNT_ID and ECOBUILD_CF_API_TOKEN in "
                    "`.env`, then restart the app. Never paste the API token "
                    "into chat or the application form."
                )
                st.code(
                    "ECOBUILD_IMAGE_PROVIDER=cloudflare\n"
                    "ECOBUILD_CF_ACCOUNT_ID=your_32_character_account_id\n"
                    "ECOBUILD_CF_API_TOKEN=your_workers_ai_token\n"
                    f"ECOBUILD_CF_MODEL={model_name}",
                    language=None,
                )
                st.markdown(
                    "Use your Cloudflare **Account ID** (not your email "
                    "address) and "
                    "[create a Workers AI API token](https://developers.cloudflare.com/workers-ai/get-started/rest-api/). "
                    "The free plan has a daily allocation; usage above it "
                    "requires upgrading."
                )
            elif account_error:
                st.error(account_error)

            if st.button(
                "Generate Eco Design",
                type="primary",
                disabled=(
                    not account_id or not token or not consent or bool(account_error)
                ),
            ):
                try:
                    prompt = _image_generation_prompt(description, preferences)
                    with st.spinner(
                        f"Generating a concept with {model_name} via Cloudflare..."
                    ):
                        st.session_state["generated_image"] = (
                            generate_cloudflare_image(
                                original,
                                prompt,
                                account_id=account_id,
                                token=token,
                                model=model_name,
                            )
                        )
                    st.success(
                        "Visual concept generated. It is not a verified plan."
                    )
                except ImageProviderConfigurationError as error:
                    st.error(str(error))
                except ImageProviderRequestError as error:
                    st.error(str(error))
        elif image_provider == "pix2pix":
            st.caption(
                "Local Pix2Pix preprocessing resizes and normalizes the image "
                "to 256 × 256. Preferences do not condition this model."
            )
            try:
                checkpoint, net_g, ngf, norm, use_dropout = _generator_settings()
            except ValueError:
                st.error(
                    "Invalid Pix2Pix settings in `.env`. ECOBUILD_NGF must be "
                    "an integer; check the checkpoint and generator configuration."
                )
                return
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not checkpoint.is_file():
                st.info(
                    "Local Pix2Pix mode requires a trained architectural "
                    "sketch-to-design checkpoint. No compatible checkpoint "
                    "is present. Use Cloudflare mode or provide trained "
                    "weights."
                )
                st.code(_display_checkpoint_path(checkpoint), language=None)
            elif st.button("Generate Eco Design", type="primary"):
                try:
                    with st.spinner(
                        f"Loading Pix2Pix on {device} and creating a concept..."
                    ):
                        generator = _cached_generator(
                            str(checkpoint),
                            str(device),
                            net_g,
                            ngf,
                            norm,
                            use_dropout,
                        )
                        st.session_state["generated_image"] = generate_image(
                            original, generator, device
                        )
                    st.success("Concept generated. It is not a verified plan.")
                except FileNotFoundError:
                    st.error(
                        "The configured checkpoint is missing. Verify its path "
                        "and model epoch in `.env`."
                    )
                except (CheckpointError, RuntimeError, ValueError) as error:
                    st.error(f"Could not run Pix2Pix inference: {error}")
        else:
            st.error(
                "Unsupported image provider. Set ECOBUILD_IMAGE_PROVIDER to "
                "`cloudflare` or `pix2pix` in `.env`."
            )

    generated = st.session_state.get("generated_image")
    if original is not None and generated is not None:
        st.subheader("Before and after")
        before, after = st.columns(2)
        with before:
            st.image(original, caption="Original image", width="stretch")
        with after:
            st.image(
                generated,
                caption="AI-generated visual concept",
                width="stretch",
            )
            output = BytesIO()
            generated.save(output, format="PNG")
            st.download_button(
                "Download generated concept",
                data=output.getvalue(),
                file_name="ecobuild-generated-concept.png",
                mime="image/png",
            )
        if image_provider == "pix2pix":
            st.info(
                "The local Pix2Pix generator is deterministic at inference "
                "time; preferences do not alter its output."
            )
        else:
            st.info(
                "This image was generated by a hosted image-edit model from "
                "your image and prompt. It is a visual concept—not a verified "
                "architectural plan or engineering recommendation."
            )


def _render_report() -> None:
    _hero(
        "Design Report",
        "A record of your concept",
        "Review or download the project description, selected preferences, and "
        "generation status without unsupported performance claims.",
    )
    report = build_design_report(
        description=st.session_state.get("project_description", ""),
        preferences=st.session_state.get("design_preferences", []),
        original_filename=st.session_state.get("design_filename"),
        generated=st.session_state.get("generated_image") is not None,
    )
    st.markdown(report)
    st.download_button(
        "Download design report",
        data=report.encode("utf-8"),
        file_name="ecobuild-design-report.md",
        mime="text/markdown",
    )


def main() -> None:
    _apply_theme()
    if st.session_state.get("page") == "Eco Assistant":
        st.session_state["page"] = "Home"
    st.session_state.pop("assistant_history", None)
    st.session_state.pop("design_explanation", None)

    with st.sidebar:
        st.markdown("## 🌿 EcoBuild GAN")
        st.caption("Sustainable Architecture Studio")
        page = st.radio(
            "Navigate",
            ["Home", "Design Studio", "Design Report"],
            key="page",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(
            "Visual concepts are exploratory—not professional engineering "
            "recommendations."
        )

    if page == "Home":
        _render_home()
    elif page == "Design Studio":
        _render_studio()
    else:
        _render_report()


if __name__ == "__main__":
    main()
