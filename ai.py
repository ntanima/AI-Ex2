import json
import math
import random
import statistics
from random import choice

import numpy as np

# *** you can change everything except the name of the class, the act function and the problem_data ***


class Table:
    def __init__(self):
        pass

    def print(self, sudoku):
        print(
            chr(0x2554)
            + 7 * chr(0x2550)
            + chr(0x2566)
            + 7 * chr(0x2550)
            + chr(0x2566)
            + 7 * chr(0x2550)
            + chr(0x2557)
        )
        for i in range(9):
            if i == 3 or i == 6:
                print(
                    chr(0x2560)
                    + 7 * chr(0x2550)
                    + chr(0x256C)
                    + 7 * chr(0x2550)
                    + chr(0x256C)
                    + 7 * chr(0x2550)
                    + chr(0x2563)
                )
            for j in range(9):
                if j % 3 == 0:
                    print(chr(0x2551), end=" ")
                print(sudoku[i][j], end=" ")
            print(chr(0x2551))
        print(
            chr(0x255A)
            + 7 * chr(0x2550)
            + chr(0x2569)
            + 7 * chr(0x2550)
            + chr(0x2569)
            + 7 * chr(0x2550)
            + chr(0x255D)
        )

    def fix_sudoku_values(self, fixedSudoku):
        for i in range(0, 9):
            for j in range(0, 9):
                if fixedSudoku[i, j] != 0:
                    fixedSudoku[i, j] = 1

        return fixedSudoku

    def block3x3(self):
        finalListOfBlocks = []
        for r in range(0, 9):
            tempList = []
            block1 = [i + 3 * ((r) % 3) for i in range(0, 3)]
            block2 = [i + 3 * math.trunc((r) / 3) for i in range(0, 3)]
            for x in block1:
                for y in block2:
                    tempList.append([x, y])
            finalListOfBlocks.append(tempList)
        return finalListOfBlocks

    def fill_block(self, sudoku, blocks):
        for block in blocks:
            for box in block:
                if sudoku[box[0], box[1]] == 0:
                    currentBlock = sudoku[
                        block[0][0] : (block[-1][0] + 1),
                        block[0][1] : (block[-1][1] + 1),
                    ]
                    sudoku[box[0], box[1]] = choice(
                        [i for i in range(1, 10) if i not in currentBlock]
                    )
        return sudoku

    def boxes_within_block(fixedSudoku, block):
        while 1:
            firstBox = random.choice(block)
            secondBox = choice([box for box in block if box is not firstBox])

            if (
                fixedSudoku[firstBox[0], firstBox[1]] != 1
                and fixedSudoku[secondBox[0], secondBox[1]] != 1
            ):
                return [firstBox, secondBox]

    def flip_boxes(sudoku, boxesToFlip):
        proposedSudoku = np.copy(sudoku)
        placeHolder = proposedSudoku[boxesToFlip[0][0], boxesToFlip[0][1]]
        proposedSudoku[boxesToFlip[0][0], boxesToFlip[0][1]] = proposedSudoku[
            boxesToFlip[1][0], boxesToFlip[1][1]
        ]
        proposedSudoku[boxesToFlip[1][0], boxesToFlip[1][1]] = placeHolder
        return proposedSudoku

    def proposed_state(self, sudoku, fixedSudoku, blocks):
        mathematics = Mathematics()
        randomBlock = random.choice(blocks)

        if mathematics.block_sum(fixedSudoku, randomBlock) > 6:
            return (sudoku, 1, 1)
        boxesToFlip = Table.boxes_within_block(fixedSudoku, randomBlock)
        proposedSudoku = Table.flip_boxes(sudoku, boxesToFlip)
        return [proposedSudoku, boxesToFlip]

    def choose_new_state(self, currentSudoku, fixedSudoku, blocks, sigma):
        mathematics = Mathematics()
        proposal = self.proposed_state(currentSudoku, fixedSudoku, blocks)
        newSudoku = proposal[0]
        boxesToCheck = proposal[1]
        currentCost = mathematics.each_row_column_cost(
            boxesToCheck[0][0], boxesToCheck[0][1], currentSudoku
        ) + mathematics.each_row_column_cost(
            boxesToCheck[1][0], boxesToCheck[1][1], currentSudoku
        )
        newCost = mathematics.each_row_column_cost(
            boxesToCheck[0][0], boxesToCheck[0][1], newSudoku
        ) + mathematics.each_row_column_cost(
            boxesToCheck[1][0], boxesToCheck[1][1], newSudoku
        )
        costDifference = newCost - currentCost
        rho = math.exp(-costDifference / sigma)
        if np.random.uniform(1, 0, 1) < rho:
            return [newSudoku, costDifference]
        return [currentSudoku, 0]


class Mathematics:
    def __init__(self):
        pass

    def each_row_column_cost(self, row, column, sudoku):
        cost = (9 - len(np.unique(sudoku[:, column]))) + (
            9 - len(np.unique(sudoku[row, :]))
        )
        return cost

    def cost_function(self, sudoku):
        cost = 0
        for i in range(0, 9):
            cost += self.each_row_column_cost(i, i, sudoku)
        return cost

    def sigma(self, sudoku, fixedSudoku, blocks):
        table = Table()
        listOfDifferences = []
        tempSudoku = sudoku
        for i in range(1, 10):
            tempSudoku = table.proposed_state(tempSudoku, fixedSudoku, blocks)[
                0
            ]
            listOfDifferences.append(self.cost_function(tempSudoku))
        return statistics.pstdev(listOfDifferences)

    def block_sum(self, sudoku, block):
        sum = 0
        for box in block:
            sum += sudoku[box[0], box[1]]
        return sum

    def iter_nums(self, fixed_sudoku):
        numberOfItterations = 0
        for i in range(0, 9):
            for j in range(0, 9):
                if fixed_sudoku[i, j] != 0:
                    numberOfItterations += 1
        return numberOfItterations


class AI:
    def __init__(self):
        pass

    def solve(self, problem):
        problem_data = json.loads(problem)
        sudoku = np.array(problem_data["sudoku"])
        table = Table()
        mathematics = Mathematics()
        num_of_solutions = 0
        while num_of_solutions == 0:
            decreaseFactor = 0.99
            stuckCount = 0
            fixedSudoku = np.copy(sudoku)
            table.print(sudoku)
            fixedSudoku = table.fix_sudoku_values(fixedSudoku)
            blocks = table.block3x3()
            tempSudoku = table.fill_block(sudoku, blocks)
            sigma = mathematics.sigma(sudoku, fixedSudoku, blocks)
            cost = mathematics.cost_function(tempSudoku)
            itterations = mathematics.iter_nums(fixedSudoku)
            if cost <= 0:
                num_of_solutions = 1

            while num_of_solutions == 0:
                previousCost = cost
                for i in range(0, itterations):
                    newState = table.choose_new_state(
                        tempSudoku, fixedSudoku, blocks, sigma
                    )
                    tempSudoku = newState[0]
                    costDiff = newState[1]
                    cost += costDiff
                    print(cost)
                    if cost <= 0:
                        num_of_solutions = 1
                        break

                sigma *= decreaseFactor
                if cost <= 0:
                    num_of_solutions = 1
                    break
                if cost >= previousCost:
                    stuckCount += 1
                else:
                    stuckCount = 0
                if stuckCount > 80:
                    sigma += 2
                if mathematics.cost_function(tempSudoku) == 0:
                    table.print(tempSudoku)
                    break
        return tempSudoku


if __name__ == "__main__":
    startingSudoku = """{
        "sudoku":
            [[0, 2, 4, 0, 0, 7, 0, 0, 0],
            [6, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 6, 8, 0, 4, 1, 5],
            [4, 3, 1, 0, 0, 5, 0, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 3, 2],
            [7, 9, 0, 0, 0, 0, 0, 6, 0],
            [2, 0, 9, 7, 1, 0, 8, 0, 0],
            [0, 4, 0, 0, 9, 3, 0, 0, 0],
            [3, 1, 0, 0, 0, 4, 7, 5, 0]]
    }"""
    # startingSudoku = """{
    #     "sudoku":
    #         [[0, 7, 9, 8, 0, 2, 0, 6, 3],
    #         [6, 0, 0, 9, 0, 0, 0, 1, 0],
    #         [8, 0, 3, 0, 7, 0, 0, 0, 2],
    #         [0, 9, 0, 0, 0, 0, 3, 7, 1],
    #         [0, 6, 8, 7, 0, 0, 0, 9, 0],
    #         [0, 3, 1, 0, 2, 0, 5, 8, 0],
    #         [2, 8, 6, 5, 0, 0, 1, 3, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [9, 0, 4, 3, 0, 0, 8, 2, 7]]
    # }"""

    ai = AI()
    table = Table()
    solution = ai.solve(startingSudoku)
    table.print(solution)
