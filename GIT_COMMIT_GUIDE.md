# Git Commit Guide for This Project

## 📚 Git Commit Best Practices

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, not functionality)
- `refactor`: Code restructuring without changing behavior
- `perf`: Performance improvements
- `test`: Adding or fixing tests
- `chore`: Maintenance tasks (dependencies, build scripts)

### Examples for This Project

```bash
# Adding new features
git commit -m "feat(rl): add CartPole policy training with REINFORCE algorithm"
git commit -m "feat(api): add /rl/action endpoint for policy inference"
git commit -m "feat(cpp): implement inferState() for RL vector input"

# Bug fixes
git commit -m "fix(jni): resolve duplicate symbol error in native library"
git commit -m "fix(cpp): handle ONNX model input name mismatch"

# Documentation
git commit -m "docs(java): add comprehensive code comments for learning"
git commit -m "docs: update README with architecture diagram"

# Refactoring
git commit -m "refactor(api): extract RLInferenceService from AIController"
git commit -m "refactor(cpp): separate RL initialization from image model init"

# Performance
git commit -m "perf(inference): add model quantization for faster CPU inference"
```

---

## 🎯 Commits to Make Right Now

Run these commands in order:

### Step 1: Stage and commit the educational comments
```bash
cd ~/Muti-LanguageAIImageClassificationSystem

# Stage the commented Java file
git add java/src/main/java/ai/controller/AIController.java

# Commit with a descriptive message
git commit -m "docs(java): add comprehensive code comments for educational reference

- Added detailed comments explaining every line of code
- Documented JNI (Java Native Interface) concepts
- Explained REST API patterns and Spark framework usage
- Added comments on CORS, multipart file uploads, error handling
- Included learning notes on Java concepts (static blocks, inner classes, etc.)
- Documented Reinforcement Learning endpoint handlers

This helps future reference and learning about:
- Java web development with Spark
- Native code integration via JNI  
- REST API design patterns
- File upload handling
- JSON serialization with Gson"
```

### Step 2: Stage and commit the model saving script
```bash
# Stage the transfer learning script
git add scripts/save_model_for_transfer_learning.py

# Commit
git commit -m "feat(scripts): add model export tool for transfer learning

- Created comprehensive Python script for saving models
- Supports ONNX model quantization (INT8) for faster inference
- Includes transfer learning template code
- Documents different model formats and their use cases
- Adds metadata export for model versioning

Features:
- List available models with --list
- Quantize models with --quantize  
- Create transfer learning templates with --create-template"
```

### Step 3: Commit any remaining RL changes
```bash
# Check what else changed
git status

# Stage all remaining changes
git add -A

# Commit the RL implementation
git commit -m "feat(rl): complete CartPole Deep RL to Java REST API pipeline

Full implementation of RL inference:
- Python: REINFORCE training script (rl/train_cartpole.py)
- ONNX: Exported policy model (models/cartpole_policy.onnx)
- C++: initializeRL() and inferState() methods
- JNI: nativeInferState() bridge function
- Java: RLInferenceService with /rl/action endpoint

Pipeline: Python Training → ONNX Export → C++ Inference → JNI → Java REST API

Tested with curl:
  curl -X POST -H 'Content-Type: application/json' \\
    -d '{\"state\": [0.0, 0.5, 0.2, 1.0]}' \\
    http://localhost:8080/rl/action"
```

### Step 4: Create a version tag
```bash
# Create an annotated tag for this milestone
git tag -a v1.0.0 -m "v1.0.0: Complete Multi-Language AI System

Features:
- Image classification via ResNet50 (ImageNet)
- RL policy inference via CartPole REINFORCE
- Java REST API with Spark framework
- C++ inference engine with ONNX Runtime
- JNI bridge for Java-C++ integration

Endpoints:
- POST /classify - Image classification
- POST /rl/action - RL policy action
- GET /health, /info, /rl/info"
```

### Step 5: Push to remote
```bash
# Push commits
git push origin main

# Push tags
git push origin --tags
```

---

## 📝 Viewing Commit History

```bash
# See commit history with graph
git log --oneline --graph --all

# See detailed commit info
git log -p -1  # Last commit with diff

# See commits by type
git log --oneline --grep="feat"
git log --oneline --grep="fix"

# See what changed in each commit
git log --stat
```

---

## 🔄 If You Need to Undo

```bash
# Undo last commit but keep changes staged
git reset --soft HEAD~1

# Undo last commit and unstage changes
git reset HEAD~1

# Completely undo last commit (⚠️ DANGEROUS - loses changes)
git reset --hard HEAD~1

# Amend last commit message
git commit --amend -m "new message"

# Add forgotten file to last commit
git add forgotten_file.py
git commit --amend --no-edit
```

---

## 🌿 Branch Strategy for Future Development

```bash
# Create feature branch for new work
git checkout -b feature/improve-accuracy

# Work on the feature...
# git add, git commit

# When done, merge back to main
git checkout main
git merge feature/improve-accuracy

# Delete the branch
git branch -d feature/improve-accuracy
```

---

## 📊 Useful Git Aliases

Add to `~/.gitconfig`:

```ini
[alias]
    st = status --short
    co = checkout
    br = branch
    ci = commit
    lg = log --oneline --graph --all --decorate
    last = log -1 HEAD --stat
    unstage = reset HEAD --
```

Then use: `git st`, `git lg`, `git last`, etc.
