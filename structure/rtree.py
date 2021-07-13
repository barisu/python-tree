from __future__ import annotations
from math import prod,log,ceil
import json

class MinimumBoundingRectangle:
    sections: tuple[tuple[float]] | None
    number_of_dimensions: int
    identifier: str | None
    child: Node | None
    node: Node | None

    def __init__(self, number_of_dimensions: int, sections: tuple[tuple[float]] | None = None, identifier: str | None = None, child: Node | None = None) -> None:
        self.sections = sections
        self.number_of_dimensions = number_of_dimensions
        self.identifier = identifier
        if child != None:
            self.child = child
            self.adjustToMBRs(child.mbrs)
        else:
            self.child = None
    # MBRが別のMBRを覆っているか判定する

    def area(self) -> float | None:
        if self.sections == None:
            return None
        return prod([b - a for a, b in self.sections])

    def isOverlaping(self, mbr: MinimumBoundingRectangle) -> bool:
        if self.sections == None or mbr.sections == None:
            return False
        for (minimum_of_section, maximum_of_section), (x, y) in zip(self.sections, mbr.sections):
            if x < minimum_of_section or maximum_of_section < y:
                return False
        return True
    # MBRが他のMBRをどの程度はみ出しているか返す

    def overflow(self, mbr: MinimumBoundingRectangle) -> float:
        if self.sections == None or mbr.sections == None:
            return 0
        area = self.area()
        sum_of_area = 0
        for (a, b), (c, d) in zip(self.sections, mbr.sections):
            sum_of_area += max(0, c - a)*area / (b - a)
            sum_of_area += max(0, b - d)*area / (b - a)
        return sum_of_area
    # 矩形を引数のMBRに対応するMBRにする

    def adjustToMBRs(self, mbrs: list[MinimumBoundingRectangle]) -> None:
        self.sections = tuple([(min([mbr.sections[i][0] for mbr in mbrs if mbr.sections != None]), max(
            [mbr.sections[i][1] for mbr in mbrs if mbr.sections != None])) for i in range(self.number_of_dimensions)])
        return

    def adjustToMBR(self, mbr: MinimumBoundingRectangle) -> None:
        if mbr.sections == None:
            return
        if self.sections == None:
            self.sections = tuple([(a, b) for a, b in mbr.sections])

        self.sections = tuple([(min(mbr.sections[i][0], self.sections[i][0]), max(
            mbr.sections[i][1], self.sections[i][1])) for i in range(self.number_of_dimensions)])

        return

    # MBRが他のMBRと等しいか判定する

    def equals(self, mbr: MinimumBoundingRectangle) -> bool:

        if self.identifier == None or mbr.identifier == None:
            return False
        if self.identifier != mbr.identifier:
            return False
        for (a, b), (c, d) in zip(self.sections, mbr.sections):
            if (a, b) != (c, d):
                return False
        return True

    def costToCover(self, mbr: MinimumBoundingRectangle) -> float | None:
        if mbr.sections == None:
            return 0
        elif self.sections == None:
            return None
        expanded_sections = []
        expanded_sections = ((min(mbr.sections[i][0], self.sections[i][0]), max(
            mbr.sections[i][1], self.sections[i][1])) for i in range(self.number_of_dimensions))
        return prod([b - a for a, b in expanded_sections]) - prod([b - a for a, b in self.sections])


class Node:
    parent_mbr: MinimumBoundingRectangle | None
    parent_node: Node
    mbrs: list[MinimumBoundingRectangle]
    maximum_node_records: int
    minimum_node_records: int
    number_of_dimensions: int

    def __init__(self, parent_node: Node  | None , parent_mbr: MinimumBoundingRectangle | None,mbrs: list[MinimumBoundingRectangle] = []) -> None:
        self.mbrs = mbrs
        self.parent_mbr = parent_mbr
        self.parent_node = parent_node
        print(self)

    def append(self, mbr: MinimumBoundingRectangle) -> None:
        self.mbrs.append(mbr)
        return

    def isLeaf(self) -> bool:
        for mbr in self.mbrs:
            if mbr.child != None:
                return False
        return True


class Rtree:

    root: Node
    maximum_node_records: int
    minimum_node_records: int
    number_of_dimensions: int
    algorithm: str

    def __init__(self, maximum_node_records: int, minimum_node_records: int = 1, number_of_dimensions: int = 1, algorithm: str = 'quadratic') -> None:
        self.root = Node(parent_node=None,parent_mbr=None)
        self.maximum_node_records = maximum_node_records
        self.minimum_node_records = max(
            minimum_node_records, maximum_node_records//2)
        self.number_of_dimensions = number_of_dimensions
        self.algorithm = algorithm

    def all(self):
        tree = {}
        stock: list[MinimumBoundingRectangle] = [mbr for mbr in self.root.mbrs]
        while len(stock) > 0:
            mbr: MinimumBoundingRectangle = stock.pop(-1)
            node: Node = mbr.child
            if node == None:
                tree[mbr] = None
                continue
            if node.isLeaf():
                tree[node] = node.mbrs
            else:
                tree[node] = {mbr: None for mbr in node.mbrs}
                stock += [mbr for mbr in node.mbrs]
        return tree

    def json(self):
        tree = {}
        stock: list[MinimumBoundingRectangle] = self.root.mbrs
        while len(stock) > 0:
            mbr: MinimumBoundingRectangle = stock.pop(-1)
            node: Node = mbr.child
            if node == None:
                tree[str(mbr)] = 'leaf'
                continue
            if node.isLeaf():
                tree[str(node)] = [str(mbr) for mbr in node.mbrs]
            else:
                tree[str(node)] = {str(mbr): 'leaf' for mbr in node.mbrs}
                stock += node.mbrs
        return json.dumps(tree)

    # 新しい矩形を挿入

    def insert(self, mbr: MinimumBoundingRectangle) -> None:
        node: Node = self.__chooseLeafToInsert(mbr, self.root)
        node.append(mbr)
        if len(node.mbrs) == self.maximum_node_records:
            group1, group2 = self.__spliteNode(node)
            node.mbrs = group1
            new_node = Node(parent_node=node.parent_node,parent_mbr=node.parent_mbr,mbrs=group2)
            self.__adjustTreeToInsert(node, splited_node=new_node)
        else:
            self.__adjustTreeToInsert(node)
        return

    # 新たな矩形を挿入した後に，その矩形に対応するように内部ノードの矩形を拡張する
    def __adjustTreeToInsert(self, node: Node, splited_node: Node | None = None) -> None:
        if self.root == node:
            print('height += 1!')
            if splited_node != None:
                new_mbr1 = MinimumBoundingRectangle(self.number_of_dimensions,child=node)
                new_mbr2 = MinimumBoundingRectangle(self.number_of_dimensions,child=splited_node)
                new_node = Node(parent_node= None,parent_mbr=None,mbrs=[new_mbr1,new_mbr2])
                node.parent_mbr = new_mbr1
                splited_node.parent_mbr = new_mbr2
                node.parent_node = new_node
                splited_node.parent_node = new_node
                self.root = new_node
            return
        node.parent_mbr.adjustToMBRs(node.mbrs)

        if splited_node != None:
            parent_mbr = MinimumBoundingRectangle(
                self.number_of_dimensions, child=splited_node)
            splited_node.parent_node = node.parent_node
            splited_node.parent_mbr = parent_mbr
            node.parent_node.append(parent_mbr)
            if len(node.parent_node.mbrs) >= self.maximum_node_records:
                mbrs1, mbrs2 = self.__spliteNode(node.parent_node)
                # node.parent_mbr.child = node
                new_node = Node(parent_node=node.parent_node,parent_mbr=node.parent_mbr,mbrs=mbrs2)
                for mbr in mbrs1:
                    mbr.child.parent_node = node.parent_node
                    mbr.child.parent_mbr = mbr
                for mbr in mbrs2:
                    mbr.child.parent_node = new_node
                    mbr.child.parent_mbr = mbr
                self.__adjustTreeToInsert(
                    node.parent_node, splited_node=new_node)

        return self.__adjustTreeToInsert(node.parent_node)

    # 矩形を挿入する葉を選択

    def __chooseLeafToInsert(self, mbr: MinimumBoundingRectangle, node: Node) -> Node:
        if node.isLeaf():
            return node
        mbrs_sorted_by_area: list[MinimumBoundingRectangle] = sorted(
            node.mbrs, key=lambda mbr_of_node: mbr_of_node.area())
        mbrs_sorted_by_overflow: list[MinimumBoundingRectangle] = sorted(
            mbrs_sorted_by_area, key=lambda mbr_of_node: mbr.overflow(mbr_of_node))
        next_node: Node = mbrs_sorted_by_overflow[0].child
        return self.__chooseLeafToInsert(mbr, next_node)

    def __spliteNode(self, node: Node) -> tuple[list[MinimumBoundingRectangle]]:
        if self.algorithm == 'quadratic':
            result = self.__quadraticSplit(node)

        return result

    def __quadraticSplit(self, node: Node) -> tuple[list[MinimumBoundingRectangle]]:

        grouped_indexes: list[bool] = [False]*len(node.mbrs)
        index1, index2 = self.__quadraticPickSeeds(node.mbrs)

        # node1 = Node(parent_node=node.parent_node,parent_mbr=node.parent_mbr)
        # node2 = Node(parent_node=node.parent_node,parent_mbr=node.parent_mbr)

        group1_mbr = MinimumBoundingRectangle(self.number_of_dimensions)
        group2_mbr = MinimumBoundingRectangle(self.number_of_dimensions)
        group1_mbr.adjustToMBR(node.mbrs[index1])
        group2_mbr.adjustToMBR(node.mbrs[index2])

        group1: list[MinimumBoundingRectangle] = [node.mbrs[index1]]
        group2: list[MinimumBoundingRectangle] = [node.mbrs[index2]]
        grouped_indexes[index1] = True
        grouped_indexes[index2] = True
        while len(group1) < self.minimum_node_records and len(group2) < self.minimum_node_records:
            mbr_index = self.__pickNext([node.mbrs[i] for i, isGrouped in enumerate(
                grouped_indexes) if not(isGrouped)], group1_mbr, group2_mbr)
            grouped_indexes[mbr_index] = True
            cost1 = group1_mbr.costToCover(node.mbrs[mbr_index])
            cost2 = group2_mbr.costToCover(node.mbrs[mbr_index])
            if cost1 > cost2:
                group2.append(node.mbrs[mbr_index])
                group2_mbr.adjustToMBR(node.mbrs[mbr_index])
            else:
                group1.append(node.mbrs[mbr_index])
                group1_mbr.adjustToMBR(node.mbrs[mbr_index])
            grouped_indexes[mbr_index] = True

        if len(group1) < self.minimum_node_records:
            group1 += [node.mbrs[index] for index,
                       isGrouped in enumerate(grouped_indexes) if not(isGrouped)]
        else:
            group2 += [node.mbrs[index] for index,
                       isGrouped in enumerate(grouped_indexes) if not(isGrouped)]
        return (group1, group2)

    def __pickNext(self, mbrs: list[MinimumBoundingRectangle], group1: MinimumBoundingRectangle, group2: MinimumBoundingRectangle) -> int:
        maximum_difference: int = 0
        the_most_different_mbr_index: MinimumBoundingRectangle | None = None
        for index, mbr in enumerate(mbrs):
            difference = abs(group1.costToCover(mbr) - group2.costToCover(mbr))
            if difference >= maximum_difference:
                maximum_difference = difference
                the_most_different_mbr_index = index
        return the_most_different_mbr_index

    def __quadraticPickSeeds(self, mbrs: list[MinimumBoundingRectangle]) -> tuple[int]:
        num: int = len(mbrs)
        worst_pairs = []
        worst_wasted_area = 0
        for i in range(num):
            for j in range(i+1, num):
                cover: MinimumBoundingRectangle = MinimumBoundingRectangle(
                    number_of_dimensions=self.number_of_dimensions)
                cover.adjustToMBRs([mbrs[i], mbrs[j]])
                wasted_area = cover.area() - mbrs[i].area() - mbrs[j].area()
                if wasted_area >= worst_wasted_area:
                    worst_pairs = (i, j)
                    worst_wasted_area = wasted_area
        return worst_pairs

    def __pickSeeds(self, mbrs: list[MinimumBoundingRectangle]):
        the_highest_low_by_dimensions: list[float] = [max(
            mbrs, lambda mbr: mbr.sections[i][0]) for i in range(self.number_of_dimensions)]
        the_lowest_high_by_dimensions: list[float] = [min(
            mbrs, lambda mbr: mbr.sections[i][1]) for i in range(self.number_of_dimensions)]
        separations: list[float] = [
            h - l for h, l in zip(the_lowest_high_by_dimensions, the_highest_low_by_dimensions)]

        return

    # あるMBRを覆うMBRを持つノードのの集合を返す

    def search(self, mbr: MinimumBoundingRectangle, node: Node = None) -> set[Node]:
        if node == None:
            node = self.root
        overlapping_mbrs: set = set()
        # nodeが葉かどうか判定して，葉であればMBRを入れた集合を返す
        if node.isLeaf():
            # 葉ノード内部のMBRを検索して検索対象を覆っているものを選ぶ
            for internal_mbr in node.mbrs:
                if internal_mbr.isOverlaping(mbr):
                    overlapping_mbrs.add(internal_mbr)
        else:
            # 葉ノードではないノードに対する処理
            # nodeが持っているMBRを一つずつ取り出す
            for internal_mbr in node.mbrs:
                if internal_mbr.isOverlaping(mbr):
                    overlapping_mbrs = overlapping_mbrs | self.search(
                        mbr, node=internal_mbr.child)
        print(node.mbrs)
        return overlapping_mbrs

    def delete(self, mbr: MinimumBoundingRectangle):
        target = self.__findLeafToDelete(mbr, self.root)
        if target == None:
            raise IndexError
        leaf, internal_index = target
        deleted_internal_mbr = leaf.mbrs[internal_index].pop(internal_index)
        del deleted_internal_mbr
        self.__codenseTree(leaf)
        return

    def __findLeafToDelete(self, mbr: MinimumBoundingRectangle, node: Node) -> tuple[Node | int] | None:
        if node.isLeaf():
            # ノードが葉である時
            for internal_index, internal_mbr in enumerate(node.mbrs):
                if mbr.equals(internal_mbr):
                    return (node, internal_index)
        else:
            # ノードが内部ノードであるとき
            for internal_index, internal_mbr in enumerate(node.mbrs):
                if internal_mbr.isOverlaping(mbr):
                    result = self.__findLeafToDelete(
                        mbr, node=internal_mbr.child)
                    if result != None:
                        return result
        return None

    # MBRの削除による矩形の縮小などを行う
    def __codenseTree(self, node: Node, eliminated_nodes: set[Node] = set()) -> None:
        if self.root == node:
            for node in eliminated_nodes:
                for mbr in node.mbrs:
                    self.insert(mbr)
        else:
            if len(node.mbrs) < self.minimum_node_records:
                parent_mbr_index: int = next([i for i, mbr in enumerate(
                    node.parent_mbr) if node.parent_mbr.equals(mbr)])
                deleted_internal_mbr = node.parent_node.pop(parent_mbr_index)
                del deleted_internal_mbr
                self.__codenseTree(
                    node.parent_node, eliminated_nodes=node.mbrs)

            else:
                node.parent_mbr.adjust(self.tree[node])
                self.__codenseTree(
                    node.parent_node, eliminated_nodes=eliminated_nodes)

        return

    # TSGに基づく整理を行う
    def bulkLoad(self,rectangles: list[MinimumBoundingRectangle]):
        sorted_rectangles: list[list[MinimumBoundingRectangle]] = []
        for i in range(self.number_of_dimensions):
            for j in range(2):
                sorted_rectangles.append(sorted(rectangles,key=lambda rectangle:rectangle.sections[i][j]))
        height: float = max(0,ceil(log(self.maximum_node_records,len(rectangles))))
        return self.bulkLoadChunk(sorted_rectangles,height)

    def bulkLoadChunk(self,sorted_rectangles: list[list[MinimumBoundingRectangle]], height: float):
        if height == 0:
            return self.buildLeafNode(sorted_rectangles[0])
        else :
            # 現在見ている木の葉に入るレコートの数
            m = self.maximum_node_records**height
            partitions = self.tgsPartitions(sorted_rectangles,m)
            new_sorted_rectangles = []
            for partition in partitions:
                new_sorted_rectangles.append(self.bulkLoadChunk(partition,height-1))
        return self.buildNonLeafNode(new_sorted_rectangles)


    def buildLeafNode(self,rectangles: list[MinimumBoundingRectangle]):
        return

    def buildNonLeafNode(self,rectangles: list[MinimumBoundingRectangle]):
        return

    def tgsPartitions(self,sorted_rectangles: list[list[MinimumBoundingRectangle]],number_of_records:int):
        if len(sorted_rectangles[0]) < number_of_records:
            return sorted_rectangles
        l,h = self.bestBinarySplit(sorted_rectangles,number_of_records)
        return (self.tgsPartitions(l,number_of_records),self.tgsPartitions(h,number_of_records))

    def bestBinarySplit(self,sorted_rectangles: list[list[MinimumBoundingRectangle]],number_of_records:int):
        number_of_partitions: int = ceil(len(sorted_rectangles[0])/number_of_records)
        minimum_cost = float('inf')
        best_sort_order = None

        for s in range(self.number_of_dimensions*2):
            f,b = self.computeBoundingBoxes(sorted_rectangles[s],number_of_records)
            for i in range(number_of_partitions-1):
                # 引数なにこれ
                cost = self.tgsCost(f,b)
                if cost < minimum_cost:
                    minimum_cost = cost
                    
        return


    def computeBoundingBoxes(self,rectangles: list[MinimumBoundingRectangle],number_of_records:int):
        return

    def tgsCost(self):
        return
