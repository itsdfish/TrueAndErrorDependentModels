module TrueAndErrorDependentModels

using TrueAndErrorModels
import TrueAndErrorModels: compute_probs

export TEDM
export compute_probs

include("model.jl")
end
