function [f1] = fmeasure(yval, pred)

for i=1:max(yval)
    if size(yval == i) == 0
        continue;
    end
    
    tp(i) = sum((pred == i) & (yval == i));
    tn(i) = sum((pred ~= i) & (yval ~= i));
    fp(i) = sum((pred ~= i) & (yval == i));
    fn(i) = sum((pred == i) & (yval ~= i));
    prec(i) = tp(i) / (tp(i) + fp(i));
    rec(i) = tp(i) / (tp(i) + fn(i));
    if (tp(i) == 0)
        f1Single(i) = 0;
    else
        f1Single(i) = 2 * prec(i) * rec(i) / (prec(i) + rec(i));
    end
end

f1 = mean(f1Single);


end
