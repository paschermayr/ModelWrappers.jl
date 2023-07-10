module ModelWrappers

############################################################################################
#Import External packages
import Base: Base, length, vec, fill, fill!, print, isapprox
import StatsBase: StatsBase, sample, sample!
import BaytesCore: BaytesCore, subset, update, generate_showvalues, generate
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
using ArgCheck: ArgCheck, @argcheck, Exception
using SimpleUnPack: SimpleUnPack, @unpack, @pack!
using Random: Random, AbstractRNG, GLOBAL_RNG

#!NOTE: These libraries are relevant for transform part
using ChainRulesCore
using LinearAlgebra: LinearAlgebra, Factorization, Cholesky, Diagonal, LowerTriangular, tril!, diag, issymmetric
using Distributions: Distributions, Distribution, logpdf
using Bijectors:
    Bijectors, Bijector, logpdf_with_trans, transform,
    TruncatedBijector, SimplexBijector, CorrBijector, PDBijector

############################################################################################
# A bunch of constants used throughout the package
"Maximum value after transformation before tagged as non-finite."
const max_val = 1e+100
"Smallest decrease allowed in the log objective results before tagged as divergent."
const min_Δ = -1e+3

function length_constrained end
function length_unconstrained end


############################################################################################
#Import
include("Core/Core.jl")
include("Models/Models.jl")

############################################################################################
#export
export
    UpdateBool,
    UpdateTrue,
    UpdateFalse,
    length_constrained,
    length_unconstrained 

end
