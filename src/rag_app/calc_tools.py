# ============================================================
# 계산 Tool: 에이전트가 암산 대신 사용합니다.
#   - calculate : 사칙연산·거듭제곱 수식 계산 (안전한 AST 평가)
# ============================================================

import ast
import operator

from langchain_core.tools import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("지원하지 않는 수식입니다.")


@tool
def calculate(expression: str) -> str:
    """수식을 정확히 계산합니다. 이자, 대출 한도, 환산 금액 등 숫자 계산은
    암산하지 말고 이 Tool을 사용하십시오.
    예) '10000000 * 0.95', '1000000 * 1355.84', '10000000 * 0.035 * 6 / 12'
    숫자와 + - * / ** % ( ) 만 사용할 수 있습니다. 쉼표나 단위는 쓰지 마십시오."""
    try:
        value = _eval(ast.parse(expression.strip(), mode="eval"))
    except ZeroDivisionError:
        return "0으로 나눌 수 없습니다."
    except Exception:
        return f"계산할 수 없는 수식입니다: {expression}"
    return f"{expression} = {value:,.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{expression} = {value:,}"
