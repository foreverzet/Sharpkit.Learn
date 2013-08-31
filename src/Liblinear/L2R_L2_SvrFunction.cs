// -----------------------------------------------------------------------
// <copyright file="L2R_L2_SvrFunction.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Liblinear
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public class L2R_L2_SvrFunction : L2R_L2_SvcFunction {


    private double p;


    public L2R_L2_SvrFunction( Problem prob, double[] C, double p ) : base(prob, C) {
        this.p = p;
    }


    public override double fun(double[] w) {
        double f = 0;
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();
        double d;


        Xv(w, z);


        for (int i = 0; i < w_size; i++)
            f += w[i] * w[i];
        f /= 2;
        for (int i = 0; i < l; i++) {
            d = z[i] - y[i];
            if (d < -p)
                f += C[i] * (d + p) * (d + p);
            else if (d > p) f += C[i] * (d - p) * (d - p);
        }


        return f;
    }


    
    public override void grad(double[] w, double[] g) {
        double[] y = prob.y;
        int l = prob.l;
        int w_size = get_nr_variable();


        sizeI = 0;
        for (int i = 0; i < l; i++) {
            double d = z[i] - y[i];


            // generate index set I
            if (d < -p) {
                z[sizeI] = C[i] * (d + p);
                I[sizeI] = i;
                sizeI++;
            } else if (d > p) {
                z[sizeI] = C[i] * (d - p);
                I[sizeI] = i;
                sizeI++;
            }


        }
        subXTv(z, g);


        for (int i = 0; i < w_size; i++)
            g[i] = w[i] + 2 * g[i];


    }

}
    }
