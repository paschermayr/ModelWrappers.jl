############################################################################################
# Import External Packages
using Test
using Random: Random, AbstractRNG, seed!

#using Soss
using LinearAlgebra
using Distributions, Bijectors, DistributionsAD
using ForwardDiff, ReverseDiff, Zygote
using ArgCheck

############################################################################################
# Import Baytes Packages
using ModelWrappers
using ModelWrappers:
    ModelWrappers,
    FlattenDefault,
    FlattenContinuous,
    FlattenAll,
    UnflattenStrict,
    UnflattenFlexible,
    _get_constraint,
    _checkparams,
    _checkfinite,
    _checkprior,
    _allparam,
    _anyparam,
    _checksampleable,
    construct_flatten,
    construct_transform,
    flatten,
    flattenAD,
    unflatten,
    unflattenAD,
    constrain,
    _get_val,
    _get_constraint,
    log_density,
    _log_density,
    log_density_and_gradient,
    _log_density_and_gradient,
    tag,
    _check,
    flatten_Symmetric,
    Symmetric_from_flatten,
    flatten_Simplex,
    Simplex_from_flatten!,
    Simplex_from_flatten

############################################################################################
# Include Files
include("TestHelper.jl")

############################################################################################
# Run Tests
@testset "All tests" begin

    include("test-flatten/flatten.jl")

    include("test-core.jl")
    include("test-flatten.jl")

    include("test-models.jl")
    include("test-tagged.jl")
    include("test-objective.jl")
    include("test-differentiation.jl")

end
