############################################################################################
# Include files

"""
$(FUNCTIONNAME)(x )
Construct a flatten function for 'x' given specifications in 'df'.

# Examples
```julia
```
"""
function construct_flatten end

"""
$(TYPEDEF)

Supertype for dispatching different types of flatten. Determines if all inputs are flattened (FlattenAll) or only continuous values (FlattenContinuous).

# Fields
$(TYPEDFIELDS)
"""
abstract type FlattenTypes end
struct FlattenAll <: FlattenTypes end
struct FlattenContinuous <: FlattenTypes end

"""
$(TYPEDEF)

Determines if unflatten returns original type or if type may change (AD friendly).

# Fields
$(TYPEDFIELDS)
"""
abstract type UnflattenTypes end
struct UnflattenStrict <: UnflattenTypes end
struct UnflattenFlexible <: UnflattenTypes end

############################################################################################
include("constrain/constrain.jl")
include("flatten/flatten.jl")

include("utility.jl")
include("checks.jl")

include("parameterinfo.jl")

############################################################################################
# Export
export FlattenTypes,
    FlattenAll,
    FlattenContinuous,
    UnflattenTypes,
    UnflattenStrict,
    UnflattenFlexible
