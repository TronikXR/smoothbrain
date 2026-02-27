---
description: How to save session brain artifacts and push to both remotes
---

## Save Brain Artifacts

After every session that produces analysis, review notes, or session artifacts:

1. Copy all session artifacts from the Antigravity brain directory to `.agent/brain/` in the smooth_brain plugin directory.
2. Commit the updated `.agent/brain/` files along with any code changes.

## Push to Both Remotes

// turbo-all

When pushing changes in the smooth_brain submodule, always push to **both** remotes. The `origin` remote is configured with dual push URLs, so a single `git push` from `F:\pinokio\api\wan.git\app\plugins\smooth_brain` will push to:

- `hoodtronik/wan2gp-smooth-brain` (origin)
- `TronikXR/smoothbrain` (origin, second push URL)

```
git push
```

If for any reason the dual push URL config is lost, manually push to both:

```
git push origin main
git push tronikxr main
```
