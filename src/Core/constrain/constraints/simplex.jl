############################################################################################
# 1. Create a new Constraint, MyConstraint <: AbstractConstraint.
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter.

# Fields
$(TYPEDFIELDS)
"""
struct Simplex{B<:Bijection} <: AbstractConstraint
    len::Int64
    bijection::B
    function Simplex(len::Integer)
        ArgCheck.@argcheck len > 0
        b = Bijection(Bijectors.SimplexBijector{true}())
        return new{typeof(b)}(len, b)
    end
end
Simplex(vec::AbstractVector) = Simplex(length(vec))

############################################################################################
#=
2. Define functions to unconstrain(constraint, val) to unconstrained domain valᵤ, and a function constrain(constraint, valᵤ) back to val.
Dimensions of val and valᵤ should be the same, flattening will be handled separately.
=#
function unconstrain(constraint::Simplex, val)
    return unconstrain(constraint.bijection, val)
end
function constrain(constraint::Simplex, valᵤ)
    return constrain(constraint.bijection, valᵤ)
end

############################################################################################
# 3. Optional - Check if check_transformer(constraint, val) works
#=
constraint = Simplex(3)
val = [.2, .3, .5]
val_u = unconstrain(constraint, val)
val_o = constrain(constraint, val_u)
check_constraint(constraint, val)
=#

############################################################################################
# 4. If Objective is used, include a method that computes logabsdet from transformation to unconstrained domain. Same syntax as in Bijectors.jl package is used, i.e., -log_abs_det_jac is returned for computations.
function log_abs_det_jac(constraint::Simplex, θ::T) where {T}
    return log_abs_det_jac(constraint.bijection, θ)
end

############################################################################################
# 5. Add _check function to check for all other peculiar things that should be tested if new releases come out and add to Test section.
function _check(
    _rng::Random.AbstractRNG,
    constraint::Simplex,
    val::Vector{R},
) where {R<:Real}
    ArgCheck.@argcheck length(val) == constraint.len
    ArgCheck.@argcheck all(val[iter] > 0.0 for iter in eachindex(val))
    return true
end

############################################################################################
# 6. Optionally - choose to only flatten k-1 parameter if Simplex is constraint
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    constraint::C,
    x::Vector{R},
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    R<:Real,
    C<:Union{Simplex, DistributionConstraint{<:Distributions.Dirichlet}, Distributions.Dirichlet, Bijectors.SimplexBijector}
}
    buffer_flat = zeros(T, length(x)-1)
    len_flat = length(x)
    len_unflat = len_flat-1
    function flatten_Simplex(x_vec::AbstractVector{R}) where {R<:Real}
        ArgCheck.@argcheck length(x_vec) == len_flat
        return fill_array!(buffer_flat, view(x_vec, 1:len_flat-1))
    end
    buffer_unflat = zeros(R, length(x))
    function unflatten_Simplex(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len_unflat
        return Simplex_from_flatten!(buffer_unflat, v)
    end
    return flatten_Simplex, unflatten_Simplex
end

function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenFlexible,
    constraint::C,
    x::Vector{R},
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    R<:Real,
    C<:Union{Simplex, DistributionConstraint{<:Distributions.Dirichlet}, Distributions.Dirichlet, Bijectors.SimplexBijector}
}
    len_flat = length(x)
    len_unflat = len_flat-1
    function flatten_Simplex_AD(x_vec::AbstractVector{R}) where {R<:Real}
        ArgCheck.@argcheck length(x_vec) == len_flat
        buffer = zeros(R, len_flat-1)
        return fill_array!(buffer, view(x_vec, 1:len_flat-1))
    end
    function unflatten_Simplex_AD(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len_unflat
        buffer = zeros(eltype(v), length(x))
        return Simplex_from_flatten!(buffer, v)
    end
    return flatten_Simplex_AD, unflatten_Simplex_AD
end

############################################################################################
#Export
export
    Simplex,
    construct_flatten,
    constrain,
    unconstrain,
    log_abs_det_jac
