############################################################################################
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::AbstractArray{R}
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    len = length(x)
    buffer_flat = zeros(T, size(vec(x)))
    function flatten_to_Real_Arr(v::AbstractArray{R}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len
        return fill_array!(buffer_flat, v)
    end
    buffer_unflat = zeros(R, size(x))
    function unflatten_to_Real_Arr(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len
        return fill_array!(buffer_unflat, v)
    end
    return flatten_to_Real_Arr, unflatten_to_Real_Arr
end
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenFlexible,
    x::AbstractArray{R}
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    len = length(x)
    function flatten_to_Real_Arr_AD(v::AbstractArray{R}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len
        return fill_array!(zeros(R, length(x)), v)
    end
    function unflatten_to_Real_Arr_AD(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len
        return fill_array!(zeros(eltype(v), size(x)), v)
    end
    return flatten_to_Real_Arr_AD, unflatten_to_Real_Arr_AD
end

############################################################################################
#Export
export
    construct_flatten
