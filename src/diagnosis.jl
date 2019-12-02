function EBFMI(Es::AbstractVector{<:AbstractScalarOrVec{<:AbstractFloat}})
    return sum(i -> (Es[i+1] - Es[i]) .^ 2, 1:length(Es)-1) ./ var(Es)
end
