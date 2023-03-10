function _get_dists(vi::VarInfo)
    mds = values(vi.metadata)
    return [md.dists[1] for md in mds]
end

function _name_variables(vi::VarInfo, dist_lengths::AbstractVector)
    vsyms = keys(vi)
    names = []
    for (vsym, dist_length) in zip(vsyms, dist_lengths)
        if dist_length==1
            name = [vsym]
            append!(names, name)
        else
            name = [DynamicPPL.VarName(Symbol(vsym, i,)) for i in 1:dist_length]
            append!(names, name)
         end
    end
    return names
end