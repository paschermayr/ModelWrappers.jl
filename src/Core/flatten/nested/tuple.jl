############################################################################################
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    x::Tuple
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
## Obtain flatten/unflatten constructor for each element
    x_vecs_and_backs = map(x) do xᵢ
        construct_flatten(T, flattentype, unflattentype, xᵢ)
    end
    _flatten, _unflatten = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
## Run flatten once to obtain dimensions of individual elements
    x_vecs = map(_flatten, x) do flat, xᵢ
         return flat(xᵢ)
    end
    lngth = map(length, x_vecs)
    sz = cumsum(lngth)
## Return flatten/unflatten for Tuple
    function construct_flatten_Tuple(x::Tuple)
        reduce(vcat,
            map(_flatten, x) do flat, xᵢ
                flat(xᵢ)
            end
        )
    end
    function construct_unflatten_Tuple(v::Union{R,AbstractVector{R}}) where {R<:Real}
        map(_unflatten, lngth, sz) do unflat, l, s
             return unflat(view(v, (s - l + 1):s))
        end
    end
    return construct_flatten_Tuple, construct_unflatten_Tuple
end

############################################################################################
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    constraint::Tuple,
    x::Tuple
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
## Obtain flatten/unflatten constructor for each element
    x_vecs_and_backs = map(x, constraint) do xᵢ, constraintᵢ
        construct_flatten(T, flattentype, unflattentype, constraintᵢ, xᵢ)
    end
    _flatten, _unflatten = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
## Run flatten once to obtain dimensions of individual elements
    x_vecs = map(_flatten, x) do flat, xᵢ
         return flat(xᵢ)
    end
    lngth = map(length, x_vecs)
    sz = cumsum(lngth)
## Return flatten/unflatten for Tuple
    function construct_flatten_Tuple(x::Tuple)
        reduce(vcat,
            map(_flatten, x) do flat, xᵢ
                flat(xᵢ)
            end
        )
    end
    function construct_unflatten_Tuple(v::Union{R,AbstractVector{R}}) where {R<:Real}
        map(_unflatten, lngth, sz) do unflat, l, s
             return unflat(view(v, (s - l + 1):s))
        end
    end
    return construct_flatten_Tuple, construct_unflatten_Tuple
end

############################################################################################
#!NOTE: If we map over collections, need to extend this to params.jl functions:
# _check, constrain, unconstrain, log_abs_det_jac

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    constraint::Tuple,
    x::Tuple
)
    return all(
            map(constraint, x) do cᵢ, xᵢ
                _check(_rng, cᵢ, xᵢ)
            end
    )
end

############################################################################################
function constrain(
    constraint::Tuple,
    θᵤ::Tuple
)
    return map(constrain, constraint, θᵤ)
end

############################################################################################
function unconstrain(
    constraint::Tuple,
    θ::Tuple
)
    return map(unconstrain, constraint, θ)
end

############################################################################################
function log_abs_det_jac(
    constraint::Tuple,
    θ::Tuple
)
    return sum(map(log_abs_det_jac, constraint, θ))
end

############################################################################################
#Export
export
    construct_flatten,
    constrain,
    unconstrain,
    log_abs_det_jac
