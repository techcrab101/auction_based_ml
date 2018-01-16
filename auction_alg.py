import numpy as np
from termcolor import colored

class Person():
    def __init__(self, ID, needs):
        self.ID = ID
        self.needs = needs
        self.object = None
        self.benefit = None

    def get_needs(self):
        return self.needs

    def get_status(self):
        x = str('ID: ' + str(self.ID) + '\n' + 
            'Needs:' + str(self.needs) + '\n' +
            'Paired Objects: ' + str(self.object) + '\n' +
            'Benefits: ' + str(self.benefit))
        return x

class Object():
    def __init__(self, ID, cost):
        self.ID = ID
        self.cost = cost

    def get_costs(self):
        return self.cost

    def get_status(self):
        x = str('ID: ' + str(self.ID) + '\n' + 
            'Costs: ' + str(self.cost))
        return x

def get_benefit(per, obj):
    return per.get_needs() - obj.get_costs()

def display_status(people, objects, pairings = None):
    print(colored('###########################', 'white'))
    if pairings is not None:
        for key in pairings:
            print(colored('Pairings:' + str(key), 'cyan'))
            print(colored('Person', 'green'))
            print(people[key[0]].get_status())
            print()
            print(colored('Object', 'yellow'))
            print(objects[key[1]].get_status())
            print()
        return

    for i in people:
        print(colored('The People', 'green'))
        stat = i.get_status()
        print(stat)
        print()

    for i in objects:
        print(colored('The Objects', 'yellow'))
        stat = i.get_status()
        print(stat)
        print()


y = 10

people = []
objects = []
pairings = {}

for i in range(y):
    people.append(Person(i, i))
    objects.append(Object(i, i))

for i in range(y):
    pairings[(i, i)] = get_benefit(people[i], objects[i])
    people[i].object = objects[i].ID

display_status(people, objects, pairings)
