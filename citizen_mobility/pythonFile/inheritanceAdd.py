class animal:
    def __init__(self, fname, lname) -> None:
        self.firstname = fname
        self.lastname = lname

    def bark(self):
        print("wannnn")


class dog(animal):
    def __init__(self, fname, lname, yearsold) -> None:
        super().__init__(fname, lname)
        self.howold = yearsold
