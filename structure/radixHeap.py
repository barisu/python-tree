from __future__ import annotations

class RadixHeap:
    # 常に最小値を返すヒープ
    # 値を格納する
    vector: list[list[int]]
    # 前回popした最小値
    last: int = 0
    # 格納している値の数
    size: int = 0
    # 格納可能な値の2進数における最大桁数
    maximum_bit_length: int = 32

    def __init__(self,maximum_bit_length: int = 32) -> None:
        self.vector = [[] for _ in range(maximum_bit_length)]
        self.maximum_bit_length = maximum_bit_length

    def push(self,val: int) -> None:
        # 新しく挿入される値は現在の最小値よりも必ず大きい
        assert(self.last <= val)
        # 挿入される値の桁数が条件を満たしている
        assert(val.bit_length() <= self.maximum_bit_length)
        # ある桁数に対応するリストへ挿入
        self.vector[(val^self.last).bit_length()].append(val)
        self.size += 1
        return

    def pop(self) -> int:
        # popする値がある
        assert(self.size > 0)
        # self.lastと等しい値がない場合self.vector[0]は空になる
        if len(self.vector[0]) == 0 :
            # 次にpopする値を含むリストを発見する
            candidates: list[int] = next(x for x in self.vector if len(x) > 0)
            # 次にpopする値を選択する
            self.last = min(candidates)
            # リストの中身を再分配する。ただし同じ配列再挿入されないことは保証される。
            for x in candidates:
                self.push(x)
            # 配列の中身の削除
            candidates.clear()
        self.size -= 1
        return  self.vector[0].pop()
