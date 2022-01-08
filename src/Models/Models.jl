############################################################################################
# External libraries
using Soss: Soss, ConditionalModel
import Soss: Soss, predict, simulate

############################################################################################
# Load sub-container
include("parameterinfo.jl")
include("modelwrapper.jl")
include("tagged.jl")
include("objective.jl")

include("_soss.jl")
############################################################################################
# Export
