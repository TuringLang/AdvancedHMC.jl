module AdvancedHMCComponentArraysExt

using AdvancedHMC: AdvancedHMC, __axes
using ComponentArrays: ComponentVecOrMat, getaxes

AdvancedHMC.__axes(r::ComponentVecOrMat) = getaxes(r)

end # module
