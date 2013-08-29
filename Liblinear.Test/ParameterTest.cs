// -----------------------------------------------------------------------
// <copyright file="ParameterTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Liblinear.Test
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    [TestClass]
    public class ParameterTest {


    private Parameter _param;


    [TestInitialize]
    public void setUp() {
        _param = new Parameter(SolverType.getById(SolverType.L2R_L1LOSS_SVC_DUAL), 100, 1e-3);
    }


    [TestMethod]
    public void testSetWeights()
    {
        Assert.IsNull(_param.weight);
        Assert.AreEqual(0, _param.getNumWeights());

        double[] weights = new double[] {0, 1, 2, 3, 4, 5};
        int[] weightLabels = new int[] {1, 1, 1, 1, 2, 3};
        _param.setWeights(weights, weightLabels);

        Assert.AreEqual(6, _param.getNumWeights());

        // assert parameter uses a copy
        weights[0]++;
        Assert.AreEqual(0, _param.getWeights()[0]);
        weightLabels[0]++;
        Assert.AreEqual(1, _param.getWeightLabels()[0]);


        weights = new double[] {0, 1, 2, 3, 4, 5};
        weightLabels = new [] {1};
        try {
            _param.setWeights(weights, weightLabels);
            Assert.Fail("ArgumentException expected");
        } catch (ArgumentException e) {
            Assert.IsTrue(e.Message.Contains("same"));
            Assert.IsTrue(e.Message.Contains("length"));
        }
    }


    [TestMethod]
    public void testGetWeights() {
        double[] weights = new double[] {0, 1, 2, 3, 4, 5};
        int[] weightLabels = new int[] {1, 1, 1, 1, 2, 3};
        _param.setWeights(weights, weightLabels);


        Assert.IsTrue(weights.SequenceEqual(_param.getWeights()));
        _param.getWeights()[0]++; // shouldn't change the parameter as we should get a copy
        Assert.IsTrue(weights.SequenceEqual(_param.getWeights()));


        Assert.IsTrue(weightLabels.SequenceEqual(_param.getWeightLabels()));
        _param.getWeightLabels()[0]++; // shouldn't change the parameter as we should get a copy
        Assert.AreEqual(1, _param.getWeightLabels()[0]);
    }


    [TestMethod]
    public void testSetC() {
        _param.setC(0.0001);
        Assert.AreEqual(0.0001, _param.getC());
        _param.setC(1);
        _param.setC(100);
        Assert.AreEqual(100, _param.getC());
        _param.setC(double.MaxValue);


        try {
            _param.setC(-1);
            Assert.Fail("ArgumentException expected");
        } catch (ArgumentException e) {
            Assert.IsTrue(e.Message.Contains("must"));
            Assert.IsTrue(e.Message.Contains("not"));
            Assert.IsTrue(e.Message.Contains("<= 0"));
        }


        try {
            _param.setC(0);
            Assert.Fail("ArgumentException expected");
        } catch (ArgumentException e) {
            Assert.IsTrue(e.Message.Contains("must"));
            Assert.IsTrue(e.Message.Contains("not"));
            Assert.IsTrue(e.Message.Contains("<= 0"));
        }
    }


    [TestMethod]
    public void testSetEps() {
        _param.setEps(0.0001);
        Assert.AreEqual(0.0001, _param.getEps());
        _param.setEps(1);
        _param.setEps(100);
        Assert.AreEqual(100, _param.getEps());
        _param.setEps(double.MaxValue);


        try {
            _param.setEps(-1);
            Assert.Fail("ArgumentException expected");
        } catch (ArgumentException e) {
            Assert.IsTrue(e.Message.Contains("must"));
            Assert.IsTrue(e.Message.Contains("not"));
            Assert.IsTrue(e.Message.Contains("<= 0"));
        }


        try {
            _param.setEps(0);
            Assert.Fail("ArgumentException expected");
        } catch (ArgumentException e)
        {
            Assert.IsTrue(e.Message.Contains("must"));
            Assert.IsTrue(e.Message.Contains("not"));
            Assert.IsTrue(e.Message.Contains("<= 0"));
        }
    }


    [TestMethod]
    public void testSetSolverType() {
        foreach (SolverType type in SolverType.values()) {
            _param.setSolverType(type);
            Assert.AreEqual(type, _param.getSolverType());
        }
        try {
            _param.setSolverType(null);
            Assert.Fail("ArgumentException expected");
        }
        catch (ArgumentException e)
        {
            Assert.IsTrue(e.Message.Contains("must"));
            Assert.IsTrue(e.Message.Contains("not"));
            Assert.IsTrue(e.Message.Contains("null"));
        }
    }
}

}
