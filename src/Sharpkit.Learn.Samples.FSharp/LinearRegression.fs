open System
open Sharpkit.Learn.LinearModel

// MathNet.Numerics.Control.LinearAlgebraProvider <- (new Mkl.MklLinearAlgebraProvider() :> ILinearAlgebraProvider)

let clf = new LinearRegression()
clf.Fit(array2D [[0.0; 0.0]; [1.0; 1.0]; [2.0; 2.0]], [|0.0; 1.0; 2.0|]) |> ignore
Console.Write(clf.Coef)

