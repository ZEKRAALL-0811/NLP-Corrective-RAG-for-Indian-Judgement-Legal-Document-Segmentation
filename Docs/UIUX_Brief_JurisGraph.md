# Document 04 — UI/UX Design Brief

**Aesthetic**
Minimal, clinical, and trustworthy — dark-mode first. Think Linear's restraint crossed with a legal-research tool's need to feel precise, not playful. No gradients, no illustrations, no marketing flourishes. Every visual choice should reinforce "this system shows its work" — the grading trace and role tags are first-class UI elements, not debug output.

**Primary Color**
`#4C6FFF` (a cooler, more restrained blue-indigo than a typical SaaS purple — reads as precise/analytical rather than consumer-playful)

**Background Color**
`#0B0D12` (near-black, slightly cool rather than pure #000 — easier on the eyes for long reading sessions over judgment text)

**Text Color**
`#E7E9EE` primary text, `#9AA1AE` secondary/muted text (timestamps, metadata labels, role tags in their unselected state)

**Accent / CTA Color**
`#4C6FFF` for primary actions (Ask, Upload). Status colors are functional, not decorative — reuse these consistently everywhere a grade appears:
- Correct → `#3DD68C` (green)
- Ambiguous → `#F5B841` (amber)
- Incorrect / discarded → `#63687A` (muted grey, not red — a discarded chunk isn't an error, it's the system working correctly)
- Verification failed / disclaimer → `#F0654B` (red, reserved only for the DeepEval-failed state, so it stays meaningful when it does appear)

**Font**
Inter for all UI text (clean, neutral, doesn't compete with dense legal text). JetBrains Mono or Geist Mono for confidence scores, role metadata tags, and any raw JSON/trace output — monospace signals "this is verifiable data," reinforcing the trust narrative.
- Headings: 20–24px, weight 600
- Body: 14–15px, weight 400
- Metadata/labels: 12px, weight 500, uppercase, letter-spacing 0.02em

**Border Radius**
6px throughout — rounded enough to feel modern, sharp enough to feel serious. Avoid the 16–20px "friendly app" radius; this isn't a consumer product.

**Shadows**
None, or near-none. Use a 1px border (`#1E212B`) to separate cards/panels instead of drop shadows — flatter, quieter, matches the "precise instrument" feel over a "delightful app" feel.

**Dark/Light Mode**
Dark mode only for the prototype (your stated preference, and it also matches "reading dense legal text for long sessions" better than stark white). Skip building a light-mode toggle — it's scope you don't need for a course project.

**Reference Apps**
Linear (restraint, monospace metadata), Notion (calm density for the document/role summary view), Raycast (how it surfaces structured trace/debug data as a first-class UI element, not an afterthought)

**Key UI Patterns**
- **Card**: document role-distribution summary after ingestion (one card per role, showing count)
- **Collapsible trace panel**: the grading step (Step 4 in your workflow) — collapsed by default, expandable, monospace confidence scores
- **Inline tag/chip**: role labels (e.g. "Arguments of Petitioner") wherever a chunk or query-mapping is shown — color-coded by role, not by grade
- **Badge**: verification status on the final answer ("✓ Verified" / "⚠ Unverified")
- **No modals** for the core flow — everything should happen inline in a single scrolling column (upload → query → trace → answer) so the user never loses context mid-flow
- **Sidebar**: optional, only if you add multi-document/session history later; not needed for the single-document prototype

**Mobile**
Not a priority — this is a research tool used at a desk with a document open, not a mobile-first product. Make it not break on a tablet-width screen (single column already helps), but don't spend build time on a mobile nav pattern.

**Accessibility**
- Text/background contrast must meet WCAG AA at minimum (`#E7E9EE` on `#0B0D12` comfortably clears this)
- Never rely on the grade color alone — always pair Correct/Ambiguous/Incorrect with the text label, not just the colored dot, since color-only status indicators fail for colorblind users
- Minimum 14px body text; don't shrink metadata below 12px
- Focus states on the query input and buttons must be visible (a 2px outline in the primary color is enough) — Streamlit handles most of this by default, but check any custom components
