from typing import Annotated
from cumm.inliner.sympy_codegen import sigmoid, tvarray_math_expr, VectorSymOperator, Scalar, Vector, VectorExpr
import sympy


class Cov2dInvOp(VectorSymOperator):
    def __init__(self):
        super().__init__()

    def forward(self, a: Annotated[sympy.Symbol, Scalar()], 
                b: Annotated[sympy.Symbol, Scalar()],
                c: Annotated[sympy.Symbol, Scalar()]) -> dict[str, VectorExpr | sympy.Expr]:
        det = a * c - b * b
        det_inv = 1.0 / det 

        
        return {
            "a_inv": c * det_inv,
            "b_inv": -b * det_inv,
            "c_inv": a * det_inv,
        }

class Cov2dInvWithCompOp(VectorSymOperator):
    def __init__(self, lowpass_filter: float = 0.3):
        super().__init__()
        self.lowpass_filter = lowpass_filter

    def forward(self, a: sympy.Symbol, 
                b: sympy.Symbol,
                c: sympy.Symbol,
                lf: sympy.Symbol) -> dict[str, VectorExpr | sympy.Expr]:
        det = a * c - b * b
        a_l = a + lf
        c_l = c + lf
        det_l = a_l * c_l - b * b
        comp = det / det_l
        det_inv = 1.0 / det_l 

        
        return {
            "a_inv": c_l * det_inv,
            "b_inv": -b * det_inv,
            "c_inv": a_l * det_inv,
            "comp": comp
        }

class Cov2dInvWithCompOpV2(VectorSymOperator):
    def __init__(self, lowpass_filter: float = 0.3):
        super().__init__()
        self.lowpass_filter = lowpass_filter

    def forward(self, a: sympy.Symbol, 
                b: sympy.Symbol,
                c: sympy.Symbol,
                lf: sympy.Symbol) -> dict[str, VectorExpr | sympy.Expr]:
        det_l = a * c - b * b
        a_l = a - lf
        c_l = c - lf
        det = a_l * c_l - b * b
        comp = det / det_l
        det_inv = 1.0 / det_l

        
        return {
            "a_inv": c * det_inv,
            "b_inv": -b * det_inv,
            "c_inv": a * det_inv,
            "comp": comp
        }


def _main():
    op = Cov2dInvOp().build()
    for k, v in op.name_to_res_sym.items():
        print(k, v.sym)

    gdict = {
        "a_inv": "da_inv",
        "b_inv": "db_inv",
        "c_inv": "dc_inv",

    }
    for k in op.name_to_sym.keys():
        print(k, op.generate_gradients_symbols(k, gdict)[0])
    # print(op.name_to_res_sym)
    op = Cov2dInvWithCompOpV2().build()
    gdict = {
        "a_inv": "da_inv",
        "b_inv": "db_inv",
        "c_inv": "dc_inv",
        "comp": "dcomp"

    }
    gdict = {
        "a_inv": None,
        "b_inv": None,
        "c_inv": None,
        "comp": "dcomp"

    }


    for k in op.name_to_sym.keys():
        print(k, (op.generate_gradients_symbols(k, gdict)[0]))

if __name__ == "__main__":
    _main()