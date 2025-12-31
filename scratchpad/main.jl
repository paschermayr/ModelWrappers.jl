############################################################################################
# main.jl - Development entry point with comment-based switching
############################################################################################
using Pkg

const SCRATCHPAD_DIR = @__DIR__
const UTILS_DIR = joinpath(SCRATCHPAD_DIR, "utils")

Pkg.activate(SCRATCHPAD_DIR)
println("✓ Environment: $SCRATCHPAD_DIR")

# Include utility functions
include(joinpath(UTILS_DIR, "io.jl"))

############################################################################################
# MODE SELECTION: Uncomment ONE of the following blocks
############################################################################################

# ============== OPTION 1: LOCAL DEV MODE ============== 
# Uncomment this block to use local development versions
const DESIRED_MODE = :local_dev
switch_mode() = use_local_dev()
# ======================================================

# ============== OPTION 2: REGISTRY MODE ============== 
# Uncomment this block to use registry versions
# const DESIRED_MODE = :registry
# switch_mode() = use_registry_versions()
# ======================================================

############################################################################################
# Auto-detect and switch if needed
############################################################################################

println("Desired Mode: $DESIRED_MODE")

# Check if ModelWrappers is in the correct mode
current_mode = :unknown
try
    pkg_info = Pkg.dependencies()
    for (uuid, info) in pkg_info
        if info.name == "ModelWrappers"
            current_mode = info.is_tracking_path ? :local_dev : :registry
            break
        end
    end
catch
    current_mode = :not_installed
end

println("Current Mode: $current_mode")

if current_mode == :not_installed
    println("⚠ Packages not installed in this environment yet!")
    println("Initializing with $DESIRED_MODE mode...")
    println()
    switch_mode()
    println()
    println("="^80)
    println("⚠ INITIALIZATION COMPLETE - RESTART JULIA")
    println("="^80)
    println("Run: exit() then julia then include(\"main.jl\")")
    println("="^80)
    return
end

if current_mode != DESIRED_MODE
    println("⚠ Mode mismatch - switching from $current_mode to $DESIRED_MODE...")
    println("-"^80)
    switch_mode()
    println()
    println("="^80)
    println("⚠ SWITCH COMPLETE - RESTART JULIA")
    println("="^80)
    println()
    println("The packages have been reconfigured.")
    println("To load the new versions:")
    println("  1. exit()")
    println("  2. julia")
    println("  3. include(\"main.jl\")")
    println("="^80)
    return
end

############################################################################################
# Load packages - mode is already correct
############################################################################################

using ModelWrappers
using BaytesCore


############################################################################################
# Verification
############################################################################################

function show_load_info(pkg_name, pkg_module)
    path = pathof(pkg_module)
    println("$pkg_name:")
    println("  $path")
    
    is_local = occursin("packages\\$(pkg_name).jl", path) || 
               occursin("packages/$(pkg_name).jl", path)
    
    if is_local
        println("  ✓ LOCAL DEV")
    else
        println("  • REGISTRY")
    end
    println()
end

show_load_info("ModelWrappers", ModelWrappers)
show_load_info("BaytesCore", BaytesCore)

check_package_version("ModelWrappers") #✓ ModelWrappers: [LOCAL DEV]
check_package_version("BaytesCore") 

############################################################################################
# Run tests
############################################################################################
Pkg.test()

############################################################################################
# Call Functions
############################################################################################


### Example Workflow
# using ModelWrappers
myparameter = (a = Float32(1.), b = 2, c = [3., 4.], d = [5, 6])
reconstruct = ReConstructor(myparameter)
vals_vec = flatten(reconstruct, myparameter) #Vector{Float64} with 3 elements (1., 3., 4.)
vals = unflatten(reconstruct, vals_vec) #(a = 1.0f0, b = 2, c = [3.0, 4.0], d = [5, 6])

flattendefault = FlattenDefault(; output = Float16, flattentype = FlattenAll())
reconstruct = ReConstructor(flattendefault, myparameter)
vals_vec = flatten(reconstruct, myparameter) #Vector{Float16} with 6 elements (1., 2., 3., 4., 5., 6.)
vals = unflatten(reconstruct, vals_vec) #(a = 1.0f0, b = 2, c = [3.0, 4.0], d = [5, 6])

Pkg.add("BenchmarkTools")
using BenchmarkTools
myparameter2 = (a = Float32(1.), b = 2, c = [3., 4.], d = [5, 6], e = randn(1000), f = rand(1:2, 1000), g = randn(1000, 2))
reconstruct = ReConstructor(myparameter2)
vals_vec = flatten(reconstruct, myparameter2)
vals_vec #Vector{Float64} with 3003 element
@btime unflatten($reconstruct, $vals_vec)   # 419.095 ns (0 allocations: 0 bytes)
@btime flatten($reconstruct, $myparameter2) # 3.475 μs (8 allocations: 39.83 KiB)





Pkg.add("Distributions")
using Distributions, Random
_rng = MersenneTwister(2)
myparameter4 = (μ = Param(Normal(), 0.0, ), σ = Param(Gamma(), 1.0, ))
mymodel = ModelWrapper(myparameter4)

Pkg.add("UnPack")
using UnPack
#Create Model and data
myparameter4 = (μ = Param(Normal(), 0.0, ), σ = Param(Gamma(), 1.0, ))
mymodel = ModelWrapper(myparameter4)
data = randn(1000)

#Create objective for both μ and σ and define a target function for it
myobjective = Objective(mymodel, data, (:μ, :σ))
function (objective::Objective{<:ModelWrapper{BaseModel}})(θ::NamedTuple)
	@unpack data = objective
	lprior = Distributions.logpdf(Distributions.Normal(),θ.μ) + Distributions.logpdf(Distributions.Exponential(), θ.σ)
    	llik = sum(Distributions.logpdf( Distributions.Normal(θ.μ, θ.σ), data[iter] ) for iter in eachindex(data))
	return lprior + llik
end

mymodel = ModelWrapper(myparameter4)
#Flatten/Unconstrain Model parameter
vals_vec = flatten(mymodel) #Vector{Float64} with 2 elements
unconstrain(mymodel) #(μ = 0.0, σ = 0.0)
unconstrain_flatten(mymodel) #Vector{Float64} with 2 elements

#Unflatten/Constrain proposed parameter from unconstrained space
θ_proposed = randn(_rng, length(vals_vec)) #Vector{Float64} with 2 elements
ModelWrappers.unflatten(mymodel, θ_proposed) #(μ = 0.7396206598864331, σ = -0.7445071021408705)
unflatten_constrain(mymodel, θ_proposed) #(μ = 0.7396206598864331, σ = 0.4749683531374296)

#Replacing current model parameter with proposed parameter
mymodel.val #(μ = 0.0, σ = 1.0)
unflatten_constrain!(mymodel, θ_proposed)
mymodel.val #(μ = 0.7396206598864331, σ = 0.4749683531374296)







