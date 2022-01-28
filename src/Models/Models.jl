############################################################################################
# External libraries

############################################################################################
# Load sub-container
include("parameterinfo.jl")
include("modelwrapper.jl")
include("tagged.jl")
include("objective.jl")
#!NOTE: Remove Soss dependency from ModelWrappers because of heavy deps. Can make separate BaytesSoss later on.
#include("_soss.jl")
############################################################################################
# Export
