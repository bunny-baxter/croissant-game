class AutoRepr():
    def __repr__(self):
        s = f"{type(self).__name__}("
        first = True
        for k, v in self.__dict__.items():
            if not first:
                s += ", "
            v_str = str(v)
            if type(v) == str:
                v_str = f"\"{v_str}\""
            s += f"{k}={v_str}"
            first = False
        s += ")"
        return s

class FakeTextBlock(AutoRepr):
    def __init__(self, text):
        self.type = "text"
        self.text = text

class FakeUsage(AutoRepr):
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

class FakeSuccessMessage(AutoRepr):
    def __init__(self, text, model, tokens):
        self.content = [ FakeTextBlock(text) ]
        self.id = "0"
        self.model = model
        self.role = "assistant"
        self.stop_reason = "end_turn"
        self.type = "message"
        self.usage = FakeUsage(0, tokens)

class FakeError(AutoRepr):
    def __init__(self, message):
        self.type = "fake_error_type"
        self.message = message

class FakeErrorMessage(AutoRepr):
    def __init__(self, message):
        self.type = "error"
        self.error = FakeError(message)

ALLOWED_MESSAGE_KEYS = { "content", "role" }
ALLOWED_CONTENT_KEYS = { "type", "text" }

class FakeMessageClient():
    def __init__(self, reply_text, is_error):
        self.reply_text = reply_text
        self.is_error = is_error

    def create(self, model, max_tokens, temperature, system, messages):
        for message in messages:
            for k, v in message.items():
                assert k in ALLOWED_MESSAGE_KEYS, f"{k} is not allowed in a `messages` element"
                if k == "content":
                    for content in v:
                        for k2, v2 in content.items():
                            assert k2 in ALLOWED_CONTENT_KEYS, f"{k2} is not allowed in a `content` element"
        if self.is_error:
            return FakeErrorMessage(self.reply_text)
        else:
            return FakeSuccessMessage(self.reply_text, model, max_tokens)

class FakeAnthropicClient():
    def __init__(self, reply_text = "Hello, world!", is_error = False):
        self.messages = FakeMessageClient(reply_text, is_error)
