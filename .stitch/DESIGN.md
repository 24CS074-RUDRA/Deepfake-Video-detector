# Deepfake Lens Design System

## Product direction
A forensic review workspace for researchers and investigators examining whether a video has been synthetically manipulated. The interface should feel precise, calm, and evidence-led rather than sensational.

## Visual language
- Platform: responsive desktop-first web app with a compact mobile reading mode.
- Mood: midnight observatory, technical, restrained, high signal.
- Background: near-black navy (#07111f) with layered radial glows and fine grid texture.
- Surfaces: navy panels (#0d1b2d) with thin blue-gray borders (#1d344b).
- Primary action: electric cyan (#68e4ef), used sparingly for active controls and confidence rings.
- Real verdict: mint (#8ae6bd). Fake verdict: coral (#ff8d79). Caution: amber (#f7c873).
- Text: warm white (#f4f7fb), secondary slate (#8ca4bd), muted steel (#5e7891).
- Typography: Space Grotesk for headings and UI, IBM Plex Mono for IDs, scores, and telemetry.
- Radius: 14px for panels, 9px for controls, 999px only for status pills and progress bars.
- Elevation: quiet shadow, 0 18px 60px rgba(0, 0, 0, .25); no floating card stacks.

## Screen structure
1. Persistent top bar with brand, model run, backend health, and reset action.
2. Intro band with product title, one-line purpose, and compact evaluation facts.
3. Main evidence area: upload intake on the left; result console on the right.
4. Result console expands into verdict, confidence, frame evidence, branch attribution, and suspicion timeline.

## Interaction rules
- Dropzone accepts MP4, MOV, AVI, and WebM and mirrors the backend's validation.
- The primary action is disabled until a video is selected.
- While analyzing, preserve the selected filename and show a deterministic progress state.
- Results use the API's label, confidence, threshold, detected frames, branch contributions, and timeline without inventing measurements.
- Empty and error states stay in the same panel footprint to avoid layout jumps.
- Use color as a secondary signal; labels and values must remain legible without color perception.
