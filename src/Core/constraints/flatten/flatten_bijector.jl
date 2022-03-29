############################################################################################
function _to_bijector(b::Bijectors.Bijector)
    return b
end
function _to_bijector(d::Distributions.Distribution)
    return Bijectors.bijector(d)
end

############################################################################################
function _checkparam(
    _rng::Random.AbstractRNG,
    val::Union{R,Array{R},AbstractArray},
    constraint::Bijectors.Bijector
) where {R<:Real}
    return typeof(constraint(val)) == typeof(val) ? true : false
end

function _checkparam(_rng::Random.AbstractRNG, val, constraint::Distributions.Distribution)
    _val = sample_constraint(_rng, constraint)
    return typeof(val) == typeof(_val) && size(val) == size(_val) ? true : false
end

############################################################################################
# Implicit constraint through density
function flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    x::Union{R,Array{R}},
    constraint::Union{Distributions.Distribution,Bijectors.Bijector},
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    U<:UnflattenTypes,
    R<:Real,
}
    return flatten(T, flattentype, unflattentype, x)
end

############################################################################################
#!NOTE: Cases where we can reduce dimensionality  -> Can make use of the fact that we did define above functions in abstract terms.

############################################################################################
function flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::Matrix{R},
    constraint::C,
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real, C<:Union{Distributions.LKJ, Bijectors.CorrBijector}}
    #= !NOTES:
        Unconstrained will always be 0 everywhere except upper diagonal elements. All other entries do not matter for constrain/unconstrain.
        Constrained will always have unit variance.
    =#
    idx_upper = tag(x, true, false)
    buffer = ones(T, size(x))
    function CorrMatrix_from_vec(x_vec::Union{<:Real,AbstractVector{<:Real}})
        return Symmetric_from_flatten!(buffer, x_vec, idx_upper)
    end
    return Vector{T}(flatten_Symmetric(x, idx_upper)), CorrMatrix_from_vec
end
function flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenAD,
    x::Matrix{R},
    constraint::C,
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real,C<:Union{Distributions.LKJ, Bijectors.CorrBijector}}
    idx_upper = tag(x, true, false)
    function CorrMatrix_from_vec_AD(x_vec::Union{<:Real,AbstractVector{<:Real}})
        return Symmetric_from_flatten!(ones(eltype(x_vec), size(x)), x_vec, idx_upper)
    end
    return Vector{T}(flatten_Symmetric(x, idx_upper)), CorrMatrix_from_vec_AD
end

############################################################################################
#!TODO: Works with flatten/unflatten - but constraint/unconstraint seems to deduce wrong type for ReverseDiff from Bijector - works fine with ForwardDiff/Zygote
function flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::Matrix{R},
    constraint::C,
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real,C<:Union{Distributions.InverseWishart, Bijectors.PDBijector}}
    idx_upper = tag(x, true, true)
    buffer = zeros(T, size(x))
    function Symmetric_from_vec(x_vec::Union{<:Real,AbstractVector{<:Real}})
        return Symmetric_from_flatten!(buffer, x_vec, idx_upper)
    end
    return Vector{T}(flatten_Symmetric(x, idx_upper)), Symmetric_from_vec
end
function flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenAD,
    x::Matrix{R},
    constraint::C,
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real,C<:Union{Distributions.InverseWishart, Bijectors.PDBijector}}
    idx_upper = tag(x, true, true)
    function Symmetric_from_vec_AD(x_vec::Union{<:Real,AbstractVector{<:Real}})
        return Symmetric_from_flatten!(zeros(eltype(x_vec), dims), x_vec, idx_upper)
    end
    return Vector{T}(flatten_Symmetric(x, idx_upper)), Symmetric_from_vec_AD
end

############################################################################################
#=
function flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::Vector{R},
    constraint::C
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    R<:Real,
    C<:Union{Distributions.Dirichlet, Bijectors.SimplexBijector}
}
#= !NOTES:
    (1) Constrained's last element will sum up to 1.
    (2) Unconstrained's last element is irrelevant for Bijector{Simplex} in default method.
    (3) Consequently, we can flatten in length(x)-1 dimensions, and unflatten back to length(x) by summing up elements for length(x)'s element.
    This works for both constrained/unconstrained.
=#
    buffer = zeros(T, length(x))
    function unflatten_Simplex(x_vec)
        return Simplex_from_flatten!(buffer, x_vec)
    end
    return Vector{T}(flatten_Simplex(x)), unflatten_Simplex
end
function flatten(output::Type{T},
    flattentype::F,
    unflattentype::UnflattenAD,
    x::Vector{R},
    constraint::C
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    R<:Real,
    C<:Union{Distributions.Dirichlet, Bijectors.SimplexBijector}
}
    function unflatten_Simplex_AD(x_vec)
        buffer = zeros(eltype(x_vec), length(x))
        return Simplex_from_flatten!(buffer, x_vec)
    end
    return Vector{T}(flatten_Simplex(x)), unflatten_Simplex_AD
end
=#

############################################################################################
# Export
export flatten
