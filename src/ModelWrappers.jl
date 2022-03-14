module ModelWrappers

############################################################################################
#Import External packages
import Base: Base, length, fill, fill!
import StatsBase: StatsBase, sample, sample!
import BaytesCore: BaytesCore, subset, update
using BaytesCore:
    AbstractModelWrapper,
    AbstractObjective,
    AbstractResult,
    BaytesCore,
    Tuple_to_Namedtuple,
    UpdateBool,
    UpdateTrue,
    UpdateFalse

using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
using ArgCheck: ArgCheck, @argcheck
using UnPack: UnPack, @unpack, @pack!
using Random: Random, AbstractRNG, GLOBAL_RNG

#!NOTE: These libraries are relevant for transform part
using LinearAlgebra: LinearAlgebra, Diagonal, LowerTriangular, tril!
using Distributions: Distributions, Distribution, logpdf
using Bijectors:
    Bijectors, Bijector, logpdf_with_trans,
    TruncatedBijector, SimplexBijector, CorrBijector, PDBijector
#!NOTE: These libraries are only relevant for 'Differentiation' files - open to make separate library later on but for now easier to test changes with AD libraries included in a single library.
using ChainRulesCore, DistributionsAD, DiffResults
using ForwardDiff, ReverseDiff, Zygote

############################################################################################
# A bunch of constants used throughout the package
"Maximum value after transformation before tagged as non-finite."
const max_val = 1e+100
"Smallest decrease allowed in the log objective results before tagged as divergent."
const min_Δ = -1e+3

############################################################################################
#Import
include("Core/Core.jl")
include("Models/Models.jl")
include("Differentiation/Differentiation.jl")

############################################################################################
#export

end
