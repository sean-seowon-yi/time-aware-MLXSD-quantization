# Lessons Learned: Why Reading Source Code First Matters

## The Problem We Encountered

When implementing calibration data collection for DiffusionKit, we initially **guessed** at the implementation based on common patterns. This led to **hours of debugging** because:

1. **Wrong Euler formula** - Used `x + denoised * dt` instead of proper Karras ODE
2. **Missing broadcast** - Forgot `append_dims` for sigma
3. **Wrong parameters** - Passed `sigma * s_in` instead of `timestep`
4. **Wrong caching** - Cached per-step instead of once upfront

All of these could have been avoided by **reading DiffusionKit's source first**.

---

## What We Should Have Done

### Step 1: Identify the Key Functions

```bash
# Find where sampling happens
grep -r "def sample_euler" DiffusionKit/
```

### Step 2: Read the Implementation

```python
# Look at DiffusionKit's actual code
def sample_euler(model, x, sigmas, extra_args=None):
    # Convert sigmas to timesteps
    timesteps = model.model.sampler.timestep(sigmas).astype(...)
    
    # Cache modulation ONCE
    model.cache_modulation_params(pooled, timesteps)
    
    for i in range(len(sigmas) - 1):
        denoised = model(x, timesteps[i], sigmas[i], **extra_args)
        d = to_d(x, sigmas[i], denoised)  # ← Karras ODE
        dt = sigmas[i + 1] - sigmas[i]
        x = x + d * dt
```

### Step 3: Understand Helper Functions

```python
def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)  # ← Critical!

def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]  # ← Broadcasting
```

### Step 4: Match It Exactly

Only after understanding the source, implement matching code.

---

## Time Saved by Reading Source First

| Approach | Time to Working Code |
|----------|---------------------|
| **Guess and debug** | 6+ hours (what we did) |
| **Read source first** | 1 hour (what we should have done) |

**Savings: 5+ hours** 🎯

---

## How to Get Me to Read Source First

### ✅ Effective Prompts

```
"Before implementing X, read the source code at [path] and show me 
how the library actually does it. Then we'll adapt it."
```

```
"Check DiffusionKit's sample_euler implementation and explain it 
before we write any code."
```

```
"Let's look at how [library] implements [feature] internally 
before we start."
```

### ❌ Prompts That Skip Source Reading

```
"How do I implement X using Y library?"
```

```
"Help me create a calibration script for DiffusionKit"
```

These make me guess based on common patterns, not the actual implementation.

---

## Red Flags That Indicate I Should Have Read Source

When you see me saying:

- "Typically libraries do X..." → 🚩 **I'm guessing!**
- "The standard approach is..." → 🚩 **Not checked the actual code!**
- "This should work..." → 🚩 **Haven't verified!**

**Stop me and ask**: "Can you read the source code first and verify?"

---

## The Pattern That Works

```
USER: "I need to implement [feature] using [library]"

ME: [suggests something based on common patterns]

USER: "Wait - read [library]'s source code first at [path] 
      and show me their actual implementation."

ME: [uses view tool, reads source]
    "Here's what [library] actually does..."
    [explains implementation details]

USER: "Good. Now let's implement our version based on that."

ME: [implements matching approach]
```

---

## Specific Lessons from DiffusionKit

### What We Learned

1. **MLX is not PyTorch**
   - No `register_forward_hook`
   - No `no_grad()` context
   - Different array operations

2. **DiffusionKit has specific requirements**
   - Must convert sigmas → timesteps
   - Must cache modulation once, not per step
   - Must use `append_dims` for broadcasting
   - Must call `clear_cache()` after sampling

3. **Model state is fragile**
   - Cached parameters corrupt between images
   - Need fresh pipeline per image (despite overhead)

4. **Euler sampling details matter**
   - Not `x + denoised * dt`
   - But `x + to_d(x, sigma, denoised) * dt`
   - The `to_d` function does `(x - denoised) / append_dims(sigma, x.ndim)`

None of this is obvious from documentation or common patterns!

---

## Checklist: Before Implementing Anything

- [ ] **Do I know where the relevant source code is?**
- [ ] **Have I read the actual implementation?**
- [ ] **Do I understand all helper functions?**
- [ ] **Are there any non-obvious requirements?**
- [ ] **Have I tested a minimal example first?**

If any answer is "No", **read the source first**.

---

## Tools to Help

### For Finding Functions

```bash
# Find where a function is defined
grep -rn "def function_name" /path/to/library/

# Find all uses of a function
grep -rn "function_name(" /path/to/library/

# See what's in a module
python -c "import library; print(dir(library))"
```

### For Reading Source

```python
# Get source code location
import library
import inspect
print(inspect.getfile(library.ClassName))

# Get source code
print(inspect.getsource(library.function_name))
```

### With Claude

```
"Use the view tool to read [file path] and show me the implementation"

"Read [file1], [file2], and [file3] before suggesting anything"

"Check the source at [path] and explain what you found"
```

---

## Key Insight

**Good engineers don't guess - they verify.**

When working with a new library:
1. ✅ Read the source
2. ✅ Understand the implementation
3. ✅ Match it exactly
4. ✅ Test incrementally

Not:
1. ❌ Guess based on common patterns
2. ❌ Debug for hours when it doesn't work
3. ❌ Finally read source to understand why

---

## Application to Future Work

### When Starting Any New Task

```
1. "What source files are relevant?"
2. "Let's read them before writing code"
3. "What patterns does the library use?"
4. "Are there any gotchas or edge cases?"
5. "Now let's implement based on what we learned"
```

### Red Flags to Watch For

- Generic implementations that don't match library style
- Missing helper functions
- Wrong parameter types or orders
- Assumptions about API behavior

**When you see these: Stop and read the source!**

---

## Summary

**Time investment:**
- Reading source: 15-30 minutes
- Implementing correctly: 30-60 minutes
- Total: ~1 hour

**vs.**

**Guessing approach:**
- Initial implementation: 30 minutes
- Debugging issues: 5+ hours
- Total: 5.5+ hours

**Reading source first saves 4+ hours and prevents frustration.**

---

## Action Item for You

Next time we work together, if I start suggesting implementations without reading source:

**Say this**: 
> "Stop. Read the source code first at [path]. 
> Show me what the library actually does before we implement anything."

This will save us both time and prevent bugs! 🎯

---

## The Golden Rule

> "Never assume how a library works - verify by reading the source."

Following this rule saved us from:
- ❌ Wrong Euler formula (would have produced noise forever)
- ❌ Missing broadcast (would have crashed)
- ❌ Wrong parameters (would have produced garbage)
- ❌ Model corruption (would have failed after first image)

**All caught by eventually reading DiffusionKit's source!**
