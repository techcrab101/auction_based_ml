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

def get_max_ben(person, objects):
    ben_ID_list = map(lambda x: [person.get_benefit(x), x], list(objects.values()))
    return max(ben_ID_list, key=lambda y: y[0])

class auction_alg():
    def __init__(self, people, objects, epsilon=None):
        self.people = people
        self.objects = objects
        if epsilon is None:
            self.epsilon = 1/(len(people) + 1)
        else:
            self.epsilon = epsilon

    def display_status(self):
        print(colored('###########################', 'white'))
        for i in list(self.people.values()):
            print(colored('The People', 'green'))
            stat = i.get_status()
            print(stat)
            print()
        for i in list(self.objects.values()):
            print(colored('The Objects', 'yellow'))
            stat = i.get_status()
            print(stat)
            print()

    def check_happiness(self):
        for i in (self.people.values()):
            if not(i.happy):
                return False
        return True

    def calculate_happiness(self):
        for i in range(len(self.people)):
            best = get_max_ben(self.people[i], objects)
            if best[0] - self.epsilon <= self.people[i].benefit:
                self.people[i].max_pot_ben = self.people[i].benefit
                self.people[i].max_pot_obj = self.people[i].object
                self.people[i].happy = True
                continue
            self.people[i].max_pot_ben = best[0]
            self.people[i].max_pot_obj = best[1].ID
            self.people[i].happy = True if self.people[i].max_pot_ben == self.people[i].benefit else False  

    def perform_auction_step(self):
        unhappy_people = filter(lambda x: x.happy is False, list(self.people.values()))
        unhappy_people_ID = list(map(lambda y: y.ID, unhappy_people))
        print(len(unhappy_people_ID))

        person_i = self.people[unhappy_people_ID[0]]
        object_i = self.objects[person_i.object]

        object_j = self.objects[person_i.max_pot_obj]
        person_j = self.people[object_j.person]

        nij = person_i.get_benefit(object_j)

        cpy_objects = copy.deepcopy(self.objects)
        del cpy_objects[person_i.max_pot_obj]

        a = get_max_ben(person_i, cpy_objects)

        nij2 = person_i.get_benefit(a[1])

        xj = nij - nij2 + self.epsilon
        object_j.cost = object_j.get_costs() + xj
        person_i.set_pairing(object_j)
        person_j.set_pairing(object_i)

        object_j.person = person_i.ID
        object_i.person = person_j.ID

        self.people[person_i.ID] = person_i
        self.people[person_j.ID] = person_j
        self.objects[object_i.ID] = object_i
        self.objects[object_j.ID] = object_j
        
        # recalculate happiness measure
        self.calculate_happiness()

    def perform_auction_process(self):
        while not(self.check_happiness()):
            self.perform_auction_step()
            self.display_status()

y = 10

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

auction_alg = auction_alg(people,objects)

auction_alg.perform_auction_process()

# TODO: Test to make sure auction alg works
