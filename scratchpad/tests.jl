############################################################################################
# tests.jl - Run tests using the PACKAGE environment (not scratchpad)
# Location: D:\Patrick\OneDrive\1_Life\1_Git\Code\Julia\packages\ModelWrappers.jl\scratchpad\tests.jl
############################################################################################

using Pkg

# Get directory paths
const SCRATCHPAD_DIR = @__DIR__
const PACKAGE_ROOT = dirname(SCRATCHPAD_DIR)  # Go up one level to ModelWrappers.jl/
const TEST_DIR = joinpath(PACKAGE_ROOT, "test")

println("="^80)
println("Running Tests with Package Environment")
println("="^80)
println()

############################################################################################
# Activate PACKAGE environment (not scratchpad!)
############################################################################################

println("Package Root: $PACKAGE_ROOT")
println()

# Activate the package's own Project.toml
Pkg.activate(PACKAGE_ROOT)
println("✓ Activated Package Environment")
println("  → $(PACKAGE_ROOT)\\Project.toml")
println()

############################################################################################
# Setup dependencies
############################################################################################

# When you activate the package itself, you DON'T need to Pkg.develop it
# The package automatically uses its own src/ folder

# Handle BaytesCore - make it use local dev if available
BAYTES_PATH = joinpath(dirname(PACKAGE_ROOT), "BaytesCore.jl")
if isdir(BAYTES_PATH)
    println("Found BaytesCore at: $BAYTES_PATH")
    
    # Check if already in dev mode
    is_dev = false
    try
        pkg_info = Pkg.dependencies()
        for (uuid, info) in pkg_info
            if info.name == "BaytesCore" && info.is_tracking_path
                is_dev = true
                println("  ✓ BaytesCore already in dev mode")
                break
            end
        end
    catch
    end
    
    if !is_dev
        print("  Use local dev version of BaytesCore? [y/n]: ")
        response = readline()
        if lowercase(strip(response)) == "y"
            Pkg.develop(path=BAYTES_PATH)
            println("  ✓ BaytesCore set to local dev")
            println("  ⚠ Restart Julia for changes to take effect")
            println()
            println("After restart, run include(\"tests.jl\") again")
            return
        end
    end
    println()
end

# Instantiate to make sure all dependencies are installed
println("Ensuring all dependencies are installed...")
Pkg.instantiate()
println()

############################################################################################
# Show what we're testing
############################################################################################

println("="^80)
println("Package Information")
println("="^80)
println()

# Show package status
println("Active Project: $(Base.active_project())")
println()

println("Dependencies:")
Pkg.status()
println()

############################################################################################
# Load and verify
############################################################################################

println("="^80)
println("Loading Package")
println("="^80)
println()

# When the package environment is active, it automatically uses src/
using ModelWrappers

# Verify it's the local version
mw_path = pathof(ModelWrappers)
println("ModelWrappers loaded from:")
println("  $mw_path")
println()

if occursin(PACKAGE_ROOT, mw_path)
    println("  ✓ Using LOCAL version from package root")
else
    println("  ⚠ WARNING: Not using local version!")
    println("  Expected path to contain: $PACKAGE_ROOT")
end

# Also check BaytesCore if it's loaded
try
    using BaytesCore
    bc_path = pathof(BaytesCore)
    println()
    println("BaytesCore loaded from:")
    println("  $bc_path")
    
    if occursin("packages\\BaytesCore.jl", bc_path) || occursin("packages/BaytesCore.jl", bc_path)
        println("  ✓ Using LOCAL dev version")
    else
        println("  • Using REGISTRY version")
    end
catch
end

println()

############################################################################################
# Run Tests
############################################################################################

println("="^80)
println("Running Test Suite")
println("="^80)
println()

# Use Pkg.test() - the official way
# This runs the test/runtests.jl file
println("Running: Pkg.test()")
println()

Pkg.test()

println()
println("="^80)
println("✓ Tests Complete")
println("="^80)