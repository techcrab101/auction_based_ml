import numpy as np
import copy
from termcolor import colored

class Person():
    def __init__(self, ID, needs):
        self.ID = ID
        self.needs = needs
        self.object = None
        self.benefit = None
        self.happy = False
        self.max_pot_ben = None
        self.max_pot_obj = None

    def get_needs(self):
        return self.needs

    def get_status(self):
        x = str('ID: ' + str(self.ID) + '\n' + 
            'Needs:' + str(self.needs) + '\n' +
            'Paired Objects: ' + str(self.object) + '\n' +
            'Benefits: ' + str(self.benefit) + '\n' +
            'Happiness: ' + str(self.happy) + '\n' +
            'max potential benefit: ' + str(self.max_pot_ben) + '\n' +
            'max potential object: ' + str(self.max_pot_obj)
            )
        return x

    def get_benefit(self, obj):
        return self.get_needs() - obj.get_costs()

    def set_pairing(self, obj):
        self.benefit = self.get_benefit(obj)
        self.object = obj.ID

class Object():
    def __init__(self, ID, cost):
        self.ID = ID
        self.cost = cost
        self.person = None

    def get_costs(self):
        return self.cost

    def get_status(self):
        x = str('ID: ' + str(self.ID) + '\n' + 
            'Costs: ' + str(self.cost) + '\n' + 
            'paired person: ' + str(self.person)
            )
        return x

def display_status(people, objects):
    print(colored('###########################', 'white'))
    for i in list(people.values()):
        print(colored('The People', 'green'))
        stat = i.get_status()
        print(stat)
        print()
    for i in list(objects.values()):
        print(colored('The Objects', 'yellow'))
        stat = i.get_status()
        print(stat)
        print()

def get_max_ben(person, objects):
    return max(map(lambda x: [people[i].get_benefit(x), x], list(objects.values())), key=lambda y: y[0])

def check_happiness(people, objects):
    for i in (people.values()):
        if not(i.happy):
            return False

    return True

y = 10
epsilon = 1/(y+1)

people = {}
objects = {}

for i in range(y):
    people[i] = Person(i, i)
    objects[i] = Object(i, y - i)

for i in range(y):
    people[i].set_pairing(objects[i])
    objects[i].person = people[i].ID
    best = get_max_ben(people[i], objects)
    people[i].max_pot_ben = best[0]
    people[i].max_pot_obj = best[1].ID
    people[i].happy = True if people[i].max_pot_ben == people[i].benefit else False  

display_status(people, objects)

while not(check_happiness(people, objects)):
    unhappy_people_ID = list(map(lambda y: y.ID, filter(lambda x: x.happy is False, list(people.values()))))
    pOI = unhappy_people_ID[0]

    person_i = people[pOI]
    object_i = objects[person_i.object]

    object_j = objects[person_i.max_pot_obj]
    person_j = people[object_j.person]


    nij = person_i.get_benefit(object_j)

    cpy_objects = copy.deepcopy(objects)
    del cpy_objects[person_i.max_pot_obj]

    a = get_max_ben(person_i, cpy_objects)

    nij2 = person_i.get_benefit(a[1])

    xj = nij - nij2 + epsilon
    object_j.cost = object_j.get_costs() + xj
    person_i.set_pairing(object_j)
    person_j.set_pairing(object_i)

    object_j.person = person_i.ID
    object_i.person = person_j.ID

    people[person_i.ID] = person_i
    people[person_j.ID] = person_j
    objects[object_i.ID] = object_i
    objects[object_j.ID] = object_j
    
    # recalculate happiness measure

    for i in range(len(people)):
        best = get_max_ben(people[i], objects)
        if best[0] - epsilon <= people[i].benefit:
            people[i].max_pot_ben = people[i].benefit
            people[i].max_pot_obj = people[i].object
            people[i].happy = True
            continue
        people[i].max_pot_ben = best[0]
        people[i].max_pot_obj = best[1].ID
        people[i].happy = True if people[i].max_pot_ben == people[i].benefit else False  

    display_status(people, objects)
