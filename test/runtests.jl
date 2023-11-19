############################################################################################
# Import External Packages
using Test
using Random: Random, AbstractRNG, seed!

#using Soss
using LinearAlgebra
using Distributions, Bijectors#, DistributionsAD
using ForwardDiff, ReverseDiff, Zygote, Enzyme
using ArgCheck

############################################################################################
# Import Baytes Packages
#include("D:/OneDrive/1_Life/1_Git/0_Dev/Julia/modules/ModelWrappers.jl/src/ModelWrappers.jl")
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
    flatten,
    flattenAD,
    unflatten,
    unflattenAD,
    constrain,
    _get_val,
    _get_constraint,
    tag,
    _check,
    flatten_Symmetric,
    Symmetric_from_flatten,
    flatten_Simplex,
    Simplex_from_flatten!,
    Simplex_from_flatten

import ModelWrappers: simulate

############################################################################################
# Include Files
include("TestHelper.jl");

############################################################################################
# Run Tests
@testset "All tests" begin

    include("test-flatten/flatten.jl")

    include("test-core.jl")
    include("test-flatten.jl")
    include("test-bijector.jl")

    include("test-models.jl")
    include("test-tagged.jl")
    include("test-objective.jl")

end
