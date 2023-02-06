############################################################################################
# 1. Create a new Constraint, MyConstraint <: AbstractConstraint.
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter.

# Fields
$(TYPEDFIELDS)
"""
struct CovarianceMatrix{B<:Bijection} <: AbstractConstraint
    bijection::B
    function CovarianceMatrix()
        b = Bijection(Bijectors.PDBijector())
        return new{typeof(b)}(b)
    end
end

############################################################################################
#=
2. Define functions to unconstrain(constraint, val) to unconstrained domain valᵤ, and a function constrain(constraint, valᵤ) back to val.
Dimensions of val and valᵤ should be the same, flattening will be handled separately.
=#
function unconstrain(constraint::CovarianceMatrix, val)
    return unconstrain(constraint.bijection, val)
end
function constrain(constraint::CovarianceMatrix, valᵤ)
    return constrain(constraint.bijection, valᵤ)
end

############################################################################################
# 3. Optional - Check if check_transformer(constraint, val) works
#=
b = Bijectors.PDBijector()
constraint = CovarianceMatrix()
val = [4. .8 ; .8 3.]
val_u = Bijectors.transform(b, val)
val_o = Bijectors.transform(inverse(b), val_u)
val_u = unconstrain(constraint, val)
val_o = constrain(constraint, val_u)
check_constraint(constraint, val)
=#

############################################################################################
# 4. If Objective is used, include a method that computes logabsdet from transformation to unconstrained domain. Same syntax as in Bijectors.jl package is used, i.e., -log_abs_det_jac is returned for computations.
function log_abs_det_jac(constraint::CovarianceMatrix, θ::T) where {T}
    return log_abs_det_jac(constraint.bijection, θ)
end

############################################################################################
# 5. Add _check function to check for all other peculiar things that should be tested if new releases come out and add to Test section.
function _check(
    _rng::Random.AbstractRNG,
    constraint::CovarianceMatrix,
    val::Matrix{R},
) where {R<:Real}
    ArgCheck.@argcheck LinearAlgebra.issymmetric(val)
    return true
end

############################################################################################
# 6. Optionally - choose to only flatten upper non-diagonal parameter if Correlationmatrix is constraint
#!TODO: Works with flatten/unflatten - but constraint/unconstraint seems to deduce wrong type for ReverseDiff from Bijector - works fine with ForwardDiff/Zygote
#!NOTE: Bijectors map to lower triangular matrix while most AD libraries evaluate upper triangular matrices.
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    constraint::C,
    x::Matrix{R},
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    R<:Real,
    C<:Union{CovarianceMatrix, DistributionConstraint{<:Distributions.InverseWishart}, Distributions.InverseWishart, Bijectors.PDBijector}
}
    #!NOTE: PDBijector seems to unconstrain to a Lower Diagonal Matrix
    idx_upper = tag(x, false, true)
    len = length(x)
    len_unflat = sum(idx_upper)
    function flatten_CovMatrix(x::AbstractMatrix{R}) where {R<:Real}
        ArgCheck.@argcheck length(x) == len
        return Vector{T}(flatten_Symmetric(x, idx_upper))
    end
    buffer_unflat = zeros(R, size(x))
    function CovMatrix_from_vec(v::Union{<:Real,AbstractVector{<:Real}})
        ArgCheck.@argcheck length(v) == len_unflat
        return Symmetric_from_flatten!(buffer_unflat, v, idx_upper)
    end
    return flatten_CovMatrix, CovMatrix_from_vec
end
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenFlexible,
    constraint::C,
    x::Matrix{R},
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    R<:Real,
    C<:Union{CovarianceMatrix, DistributionConstraint{<:Distributions.InverseWishart}, Distributions.InverseWishart, Bijectors.PDBijector}
}
    #!NOTE: PDBijector seems to unconstrain to a Lower Diagonal Matrix
    idx_upper = tag(x, false, true)
    len = length(x)
    len_unflat = sum(idx_upper)
    function flatten_CovMatrix_AD(x::AbstractMatrix{R}) where {R<:Real}
        ArgCheck.@argcheck length(x) == len
        return Vector{R}(flatten_Symmetric(x, idx_upper))
    end
    dims = size(x)
    function CovMatrix_from_vecAD(v::Union{<:Real,AbstractVector{<:Real}})
        ArgCheck.@argcheck length(v) == len_unflat
        buffer_unflat = zeros(eltype(v), dims)
        return Symmetric_from_flatten!(buffer_unflat, v, idx_upper)
    end
    return flatten_CovMatrix_AD, CovMatrix_from_vecAD
end

############################################################################################
#Export
export
    CovarianceMatrix,
    construct_flatten,
    constrain,
    unconstrain,
    log_abs_det_jac
