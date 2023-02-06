############################################################################################
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    x::AbstractArray
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
## Obtain flatten/unflatten constructor for each element
    x_vecs_and_backs = map(x) do xᵢ
        construct_flatten(T, flattentype, unflattentype, xᵢ)
    end
    _flatten, _unflatten = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
## Run flatten once to obtain dimensions of individual elements
    x_vecs = map(iter -> _flatten[iter](x[iter]), eachindex(x))
    lngth = map(length, x_vecs)
    sz = cumsum(lngth)
## Return flatten/unflatten for AbstractArray
    function construct_flatten_AbstractArr(x::AbstractArray)
        #!NOTE: reduce(vcat) seems to be faster for Array than tuple, thus speedincrease
        reduce(vcat,
            map(iter -> _flatten[iter](x[iter]), eachindex(x))
            )
    end
    function construct_unflatten_AbstractArr(v::Union{R,AbstractVector{R}}) where {R<:Real}
        return [
            _unflatten[n](view(v, (sz[n] - lngth[n] + 1):sz[n])) for n in eachindex(x)
        ]
    end
    return construct_flatten_AbstractArr, construct_unflatten_AbstractArr
end

############################################################################################
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    constraint::AbstractArray,
    x::AbstractArray
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
## Obtain flatten/unflatten constructor for each element
    x_vecs_and_backs = map(x, constraint) do xᵢ, constraintᵢ
        construct_flatten(T, flattentype, unflattentype, constraintᵢ, xᵢ)
    end
    _flatten, _unflatten = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
## Run flatten once to obtain dimensions of individual elements
    x_vecs = map(iter -> _flatten[iter](x[iter]), eachindex(x))
    lngth = map(length, x_vecs)
    sz = cumsum(lngth)
## Return flatten/unflatten for AbstractArray
    function construct_flatten_AbstractArr(x::AbstractArray)
        #!NOTE: reduce(vcat) seems to be faster for Array than tuple, thus speedincrease
        reduce(vcat,
            map(iter -> _flatten[iter](x[iter]), eachindex(x))
            )
    end
    function construct_unflatten_AbstractArr(v::Union{R,AbstractVector{R}}) where {R<:Real}
        return [
            _unflatten[n](view(v, (sz[n] - lngth[n] + 1):sz[n])) for n in eachindex(x)
        ]
    end
    return construct_flatten_AbstractArr, construct_unflatten_AbstractArr
end

############################################################################################
#!NOTE: If we map over collections, need to extend this to params.jl functions:
# _check, constrain, unconstrain, log_abs_det_jac

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    constraint::AbstractArray,
    x::AbstractArray
)
    return all(
            map(constraint, x) do cᵢ, xᵢ
                _check(_rng, cᵢ, xᵢ)
            end
    )
end
#=
############################################################################################
function construct_transform(
    constraint::AbstractArray,
    x::AbstractArray
)
## Obtain transform/inversetransform constructor for each element
    x_transforms = map(x, constraint) do xᵢ, constraintᵢ
        construct_transform(constraintᵢ, xᵢ)
    end
    _transform, _inversetransform = first.(x_transforms), last.(x_transforms)
## Return flatten/unflatten for AbstractArray
    return _transform, _inversetransform
end
=#
############################################################################################
function constrain(
    constraint::AbstractArray,
    θᵤ::AbstractArray
)
    return map(constrain, constraint, θᵤ)
end

############################################################################################
function unconstrain(
    constraint::AbstractArray,
    θ::AbstractArray
)
    return map(unconstrain, constraint, θ)
end

############################################################################################
function log_abs_det_jac(
    constraint::AbstractArray,
    θ::AbstractArray
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
