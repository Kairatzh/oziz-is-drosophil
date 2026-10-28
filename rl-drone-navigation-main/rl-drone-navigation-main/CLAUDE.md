# Working Rules for This Project

This project follows the phased plan in `PROJECT_SPEC.md`. These rules govern how work proceeds.

- Work through the phases in `PROJECT_SPEC.md` strictly in order.
- Never skip ahead to a later phase without the user's explicit approval.
- After each phase, STOP and explain what was built and why it matters for RL. This is the user's first ML project — explain concepts clearly, don't assume prior ML knowledge.
- Write tests for a phase before moving on to the next one.
- All plots must use `plt.savefig()` followed by `plt.close('all')`. NEVER call `plt.show()` — it blocks indefinitely on Windows.
- Never add `Co-authored-by` trailers to git commits.
- Suggest a commit at the end of each validated phase, with a clear descriptive message.
- If training seems stuck, or a phase is taking very long with no clear path forward: STOP, summarize hypotheses tried and their results, and propose one minimal next test. Do not iterate indefinitely.
- Default to CPU training. Do not attempt AMD GPU (ROCm) acceleration without first discussing it with the user.
- If anything is ambiguous, ask the user rather than assuming.
