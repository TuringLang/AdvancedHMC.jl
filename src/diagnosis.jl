function EBFMI(Es::AbstractVector{<:AbstractScalarOrVec{<:AbstractFloat}})
    return mean(i -> (Es[i+1] - Es[i]) .^ 2, 1:length(Es)-1) ./ var(Es)
end
