/**
 * Postinstall script: ensures arc-agi Python package and game dependencies are installed.
 * Tries multiple Python/pip binaries in order of preference.
 * Writes the working Python path to .python-bin for the bridge to use.
 */
const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const MARKER = path.join(ROOT, ".python-bin");

// Python binaries to try, in order
const candidates = [
  // Project venv at root
  path.join(ROOT, ".venv", "bin", "python3"),
  // Legacy location
  path.join(ROOT, "backend", "venv", "bin", "python3"),
  // Homebrew Python 3.12+
  "/opt/homebrew/bin/python3",
  "/usr/local/bin/python3",
  // System
  "python3",
  "python",
];

// All Python packages required by the game engine and game environments
const PACKAGES = ["arc-agi", "gymnasium", "numpy"];

function tryPython(bin) {
  try {
    // Check if binary exists and has all required packages
    const check = PACKAGES.map(p => {
      // arc-agi installs as 'arcengine' import
      const imp = p === "arc-agi" ? "arcengine" : p;
      return `import ${imp}`;
    }).join("; ");
    execSync(`${bin} -c "${check}; print('ok')"`, {
      stdio: "pipe",
      timeout: 10000,
    });
    return true;
  } catch {
    return false;
  }
}

function tryInstall(bin) {
  try {
    // Check Python version >= 3.10
    const ver = execSync(`${bin} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"`, {
      encoding: "utf-8",
      stdio: "pipe",
      timeout: 5000,
    }).trim();

    const [major, minor] = ver.split(".").map(Number);
    if (major < 3 || (major === 3 && minor < 10)) {
      return false;
    }

    // Try pip install
    const pkgList = PACKAGES.join(" ");
    console.log(`[setup] Installing ${pkgList} via ${bin} (Python ${ver})...`);
    execSync(`${bin} -m pip install ${pkgList} --quiet --user`, {
      stdio: "inherit",
      timeout: 120000,
    });
    return true;
  } catch {
    return false;
  }
}

// Step 1: Check if any candidate already has all packages
for (const bin of candidates) {
  if (tryPython(bin)) {
    console.log(`[setup] All Python dependencies already available via: ${bin}`);
    fs.writeFileSync(MARKER, bin, "utf-8");
    process.exit(0);
  }
}

// Step 2: Try installing all packages with each candidate
for (const bin of candidates) {
  if (tryInstall(bin)) {
    // Verify it worked
    if (tryPython(bin)) {
      console.log(`[setup] All Python dependencies installed successfully via: ${bin}`);
      fs.writeFileSync(MARKER, bin, "utf-8");
      process.exit(0);
    }
  }
}

console.warn("[WARN] Could not install Python dependencies automatically.");
console.warn("       The game engine requires Python 3.10+ with: " + PACKAGES.join(", "));
console.warn("       Install manually: pip3 install " + PACKAGES.join(" "));
console.warn("       Then set PYTHON_BIN in .env to your Python path.");
process.exit(0); // Don't fail npm install
