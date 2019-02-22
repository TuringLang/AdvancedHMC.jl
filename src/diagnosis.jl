function EBFMI(Es)
    return sum((Es[2:end] - Es[1:end-1]).^2) / var(Es)
end
