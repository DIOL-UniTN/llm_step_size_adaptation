import requests
import numpy as np
from time import sleep
from IOHexperimenter import IOH_logger, IOHexperimenter


GROQ_TOKEN = "YOUR_TOKEN_HERE"


def query(prompt, url, token="", model=""):
    try:
        ctx_size = int(model.split('-')[-1])
    except:
        ctx_size = 8192
    if len(prompt) // 4 > ctx_size:  # Approx 2 chars per token
        print("Prompt too long")
        new_prompt = prompt.split('```txt')[0] + '```txt\n'
        new_prompt += prompt[-ctx_size - len(new_prompt) // 4:]
        prompt = "Current".join(new_prompt.split('Current')[1:])
        print(prompt)
    data_json = {
      "messages": [
            {"role": "system", "content": "You are a powerful and intelligent AI capable of analyzing logs and performing reasoning"},
            {"role": "user", "content": prompt + "\nReply with the following structure: `Reasoning: <explanation>\nRecommended step size: <new step size>`"},
      ],
      "model": model
    }
    sleep(3)  # This avoids the possibility of exceeding the request limits

    for i in range(5):
        try:
            resp = requests.post(
                url=url,
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
                json=data_json,
            )
            print(resp.text)

            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            print(f"Request was not successful. {i}-th try")
            sleep(10)  # This avoids the possibility of exceeding the request limits
    return ""


def get_model_indications(param, log, url, token="", model=""):
    prompt = ""
    prompt += 'Q: I am running an optimization process over an unknown function.' + 'I am using a 1+1-ES to optimize the function.\n'
    prompt += 'Each row indicates an evaluation (i.e., the fitness). Our goal is to minimize the values the fitness. It is extremely important that the step size you propose is contained between 0.999 and 0.001. '
    prompt += """Here's the log:```txt\n"""
    prompt += log
    prompt += '```\n'
    prompt += f'I am currently using the following step size: {param}. Should I change it or not? Do you think that the current step size is good enough to make the process converge as soon as possible?\n'
    reply = query(prompt, url, token, model)

    if "Recommended step size" in reply:
        try:
            return float(reply.split("Recommended step size: ")[-1].split('\n')[0])
        except:
            return param
    return param


class OnePlusOne:
    def __init__(self, budget, initial_stepsize):
        self.budget = int(budget)
        self.f_opt = np.Inf
        self.x_opt = None
        self.stepsize = initial_stepsize
        self._initial_stepsize = initial_stepsize
        self.history = []

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.stepsize = self._initial_stepsize
        self.history = []
        for i in range(self.budget):
            self.new_step(func)
        return self.f_opt, self.x_opt

    def new_step(self, func):
        if self.x_opt is None:
            x = np.random.uniform(func.lowerbound, func.upperbound)
        else:
            x = self.x_opt + np.random.normal(0, self.stepsize, size=self.x_opt.shape)
        f = func(x)
        self.history.append((x, f))
        if self.x_opt is None or f < self.f_opt:
            self.f_opt = f
            self.x_opt = x
    
    @property
    def step_size(self):
        return self.stepsize


class OneFifthOnePlusOne(OnePlusOne):
    def __init__(self, budget, initial_stepsize):
        OnePlusOne.__init__(self, budget, initial_stepsize)
        self._initial_stepsize = initial_stepsize
        self.stepsizes = [initial_stepsize]
        self.bests = [(None,None)]
        self._period = 1

    def __call__(self, func):
        self.stepsizes = [self._initial_stepsize]
        self.bests = [(None,None)]
        self.f_opt = np.Inf
        self.x_opt = None
        self.stepsize = self._initial_stepsize
        self.history = []
        for i in range(0, self.budget):
            self.new_step(func)
            self.bests.append((self.x_opt, self.f_opt))
            self.update_stepsize()
        return self.f_opt, self.x_opt

    def update_stepsize(self):
        if len(self.bests) < 2 or self.bests[-2][1] is None:
            pass
        else:
            if self.bests[-1][1] < self.bests[-2][1]:
                self.stepsize = 1.5 * self.stepsize
            elif self.bests[-1][1] > self.bests[-2][1]:
                self.stepsize = (1.5 ** -0.25) * self.stepsize

    @property
    def step_size(self):
        return self.stepsize


class LLMTunedOnePlusOne(OnePlusOne):
    def __init__(self, budget, initial_stepsize, period=2):
        OnePlusOne.__init__(self, budget, initial_stepsize)
        self._initial_stepsize = initial_stepsize
        self.stepsizes = [initial_stepsize]
        self.bests = [(None,None)]
        self._period = period
        self.url = "http://127.0.0.1:8081/completion"
        self.token = ""

    def __call__(self, func):
        self.stepsizes = [self._initial_stepsize]
        self.bests = [(None,None)]
        self.f_opt = np.Inf
        self.x_opt = None
        self.stepsize = self._initial_stepsize
        self.history = []
        for i in range(0, self.budget, self._period):
            for _ in range(self._period):
                self.new_step(func)
                self.bests.append((self.x_opt, self.f_opt))
            self.update_stepsize()
        return self.f_opt, self.x_opt

    def update_stepsize(self):
        log = ""
        for i, step in enumerate(self.stepsizes):
            log += f"Current step size: {step}\n" 
            for j in range(self._period*i, self._period*(i+1)):
                x, y = self.history[j]
                log += f"{y:.2e}\n"
        self.stepsize = get_model_indications(self.stepsizes[-1], log, url=self.url, token=self.token, model=self.model)
        print("Setting stepsize to", self.stepsize)
        self.stepsizes.append(self.stepsize)

    @property
    def step_size(self):
        return self.stepsize

class MixtralTunedOnePlusOne(LLMTunedOnePlusOne):
    def __init__(self, budget, initial_stepsize, period=2):
        LLMTunedOnePlusOne.__init__(self, budget, initial_stepsize, period=period)
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.token = GROQ_TOKEN
        self.model = "mixtral-8x7b-32768"

    @property
    def step_size(self):
        return self.stepsize

class Llama70bTunedOnePlusOne(LLMTunedOnePlusOne):
    def __init__(self, budget, initial_stepsize, period=2):
        LLMTunedOnePlusOne.__init__(self, budget, initial_stepsize, period=period)
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.token = GROQ_TOKEN
        self.model = "llama2-70b-4096"

    @property
    def step_size(self):
        return self.stepsize

class Gemma7bTunedOnePlusOne(LLMTunedOnePlusOne):
    def __init__(self, budget, initial_stepsize, period=2):
        LLMTunedOnePlusOne.__init__(self, budget, initial_stepsize, period=period)
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.token = GROQ_TOKEN
        self.model = "gemma-7b-it"

    @property
    def step_size(self):
        return self.stepsize


if __name__ == "__main__":
    opts = [
        OnePlusOne(1000, 0.1),
        OneFifthOnePlusOne(1000, 0.1),
        Llama70bTunedOnePlusOne(1000, 0.1, 50),
        MixtralTunedOnePlusOne(1000, 0.1, 50),
    ]

    logger = IOH_logger(location = "data", foldername = "parameter-tracking", 
                        name = "random_search", 
                        info = "test of IOHexperimenter in python")

    logger.track_parameters(algorithm = opts, parameters = 'step_size')

    exp = IOHexperimenter()
    exp.initialize_BBOB(fids = [*range(1, 25)], dims = [2, 5, 30], iids = [1], repetitions = 10)
    exp.set_logger_location(location = "data_test", foldername = "runs")
    exp.set_parameter_tracking("step_size")
    exp(opts)
