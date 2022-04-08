############################################################################################
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter.

# Fields
$(TYPEDFIELDS)
"""
struct CorrelationMatrix <: AbstractConstraint
    function CorrelationMatrix()
        return new()
    end
end

############################################################################################
function construct_transform(constraint::CorrelationMatrix, val)
    transform = Bijectors.CorrBijector()
    return transform, Bijectors.inverse(transform)
end

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    constraint::CorrelationMatrix,
    val::Matrix{R},
) where {R<:Real}
    ArgCheck.@argcheck all(LinearAlgebra.diag(val) .== 1.0)
    return true
end

############################################################################################
#= !NOTES:
    Unconstrained will always be 0 everywhere except upper diagonal elements. All other entries do not matter for constrain/unconstrain.
    Constrained will always have unit variance.
=#
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
    C<:Union{CorrelationMatrix, Distributions.LKJ, Bijectors.CorrBijector}
}
    #!NOTE: CorrBijector seems to unconstrain to a Upper Diagonal Matrix
    idx_upper = tag(x, true, false)
    len = length(x)
    len_unflat = sum(idx_upper)
    #!NOTE: Buffer should be of type R, not T, as we want same type back afterwards
    function flatten_CorrMatrix(x::AbstractMatrix{R}) where {R<:Real}
        ArgCheck.@argcheck length(x) == len
        return Vector{T}(flatten_Symmetric(x, idx_upper))
    end
    buffer_unflat = ones(R, size(x))
    function CorrMatrix_from_vec(v::Union{<:Real,AbstractVector{<:Real}})
        ArgCheck.@argcheck length(v) == len_unflat
        return Symmetric_from_flatten!(buffer_unflat, v, idx_upper)
    end
    return flatten_CorrMatrix, CorrMatrix_from_vec
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
    C<:Union{CorrelationMatrix, Distributions.LKJ, Bijectors.CorrBijector}
}
    #!NOTE: CorrBijector seems to unconstrain to a Upper Diagonal Matrix
    idx_upper = tag(x, true, false)
    len = length(x)
    len_unflat = sum(idx_upper)
    function flatten_CorrMatrix_AD(x::AbstractMatrix{R}) where {R<:Real}
        ArgCheck.@argcheck length(x) == len
        return Vector{R}(flatten_Symmetric(x, idx_upper))
    end
    function CorrMatrix_from_vec_AD(v::Union{<:Real,AbstractVector{<:Real}})
        ArgCheck.@argcheck length(v) == len_unflat
        return Symmetric_from_flatten!(ones(eltype(v), size(x)), v, idx_upper)
    end
    return flatten_CorrMatrix_AD, CorrMatrix_from_vec_AD
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
    CorrelationMatrix,
    construct_flatten,
    construct_transform,
    constrain,
    unconstrain,
    log_abs_det_jac
