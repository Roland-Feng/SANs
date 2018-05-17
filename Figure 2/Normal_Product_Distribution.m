function densityValue = Normal_Product_Distribution(x,s1,s2)

densityValue = Modified_Bessel_Zero(abs(x)/(s1*s2))/(pi*s1*s2);

end