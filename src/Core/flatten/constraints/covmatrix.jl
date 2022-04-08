############################################################################################
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter.

# Fields
$(TYPEDFIELDS)
"""
struct CovarianceMatrix <: AbstractConstraint
    function CovarianceMatrix()
        return new()
    end
end

############################################################################################
function construct_transform(constraint::CovarianceMatrix, val)
    transform = Bijectors.PDBijector()
    return transform, Bijectors.inverse(transform)
end

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    constraint::CovarianceMatrix,
    val::Matrix{R},
) where {R<:Real}
    ArgCheck.@argcheck LinearAlgebra.issymmetric(val)
    return true
end

idx_upper = tag(zeros(3,3), false, true)
sum(idx_upper)
############################################################################################
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
    C<:Union{CovarianceMatrix, Distributions.InverseWishart, Bijectors.PDBijector}
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
    C<:Union{CovarianceMatrix, Distributions.InverseWishart, Bijectors.PDBijector}
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
#=
!NOTE: For this constraint, we use a Bijector as Transformer, so we do not need to add a new functors
1. MyTransformer <: AbstractTransformer, MyInverseTransformer <: AbstractTransformer
2. define a function construct_transform(MyConstraint, val) -> MyTransformer, MyInverseTransformer
3. overload unconstrain and log_abs_det_jac on MyTransformer.
4. overload constrain on MyInverseTransformer.
=#

############################################################################################
#Export
export
    CovarianceMatrix,
    construct_flatten,
    construct_transform,
    constrain,
    unconstrain,
    log_abs_det_jac
