############################################################################################
# 1. Create a new Constraint, MyConstraint <: AbstractConstraint.
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter.

# Fields
$(TYPEDFIELDS)
"""
struct CorrelationMatrix{B<:Bijection} <: AbstractConstraint
    bijection::B
    function CorrelationMatrix()
        b = Bijection(Bijectors.CorrBijector())
        return new{typeof(b)}(b)
    end
end

############################################################################################
#=
2. Define functions to unconstrain(constraint, val) to unconstrained domain valᵤ, and a function constrain(constraint, valᵤ) back to val.
Dimensions of val and valᵤ should be the same, flattening will be handled separately.
=#
function unconstrain(constraint::CorrelationMatrix, val)
    return unconstrain(constraint.bijection, val)
end
function constrain(constraint::CorrelationMatrix, valᵤ)
    return constrain(constraint.bijection, valᵤ)
end

############################################################################################
# 3. Optional - Check if check_transformer(constraint, val) works
#=
constraint = CorrelationMatrix()
val = [1. .2 ; .2 1.]
val_u = unconstrain(constraint, val)
val_o = constrain(constraint, val_u)
check_constraint(constraint, val)
=#

############################################################################################
# 4. If Objective is used, include a method that computes logabsdet from transformation to unconstrained domain. Same syntax as in Bijectors.jl package is used, i.e., -log_abs_det_jac is returned for computations.
function log_abs_det_jac(constraint::CorrelationMatrix, θ::T) where {T}
    return log_abs_det_jac(constraint.bijection, θ)
end

############################################################################################
# 5. Add _check function to check for all other peculiar things that should be tested if new releases come out and add to Test section.
function _check(
    _rng::Random.AbstractRNG,
    constraint::CorrelationMatrix,
    val::Matrix{R},
) where {R<:Real}
    ArgCheck.@argcheck all(LinearAlgebra.diag(val) .== 1.0)
    return true
end

############################################################################################
# 6. Optionally - choose to only flatten upper non-diagonal parameter if Correlationmatrix is constraint
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
    C<:Union{CorrelationMatrix, DistributionConstraint{<:Distributions.LKJ}, Distributions.LKJ, Bijectors.CorrBijector}
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
    C<:Union{CorrelationMatrix, DistributionConstraint{<:Distributions.LKJ},Distributions.LKJ, Bijectors.CorrBijector}
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
#Export
export
    CorrelationMatrix,
    construct_flatten,
    constrain,
    unconstrain,
    log_abs_det_jac
