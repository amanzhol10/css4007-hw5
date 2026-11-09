from helpdesk.memory import UserMemory


def test_new_thread_of_same_user_sees_preference_but_not_history(make_app):
    app = make_app()
    app.send("demo-user-a", "a1", "Please remember that I prefer replies in Kazakh. I enabled Wi-Fi and tried reconnecting.")
    result = app.send("demo-user-a", "a2", "Which language did I ask you to use? Which problem did we discuss?")
    assert "Kazakh" in result["reply"]
    assert "No previous problem" in result["reply"]
    assert any(s.startswith("user_memory[demo-user-a]") and "Kazakh" in s for s in result["sources"])
    assert not any(s.startswith("conversation_state") for s in result["sources"])


def test_other_user_does_not_see_preference(make_app):
    app = make_app()
    app.send("demo-user-a", "a1", "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.")
    result = app.send("demo-user-b", "b1", "Have I told you a preferred language or a previous problem?")
    assert "Kazakh" not in result["reply"]
    assert "No preferred language" in result["reply"]
    assert "No previous problem" in result["reply"]
    assert result["sources"] == ["user_memory[demo-user-b]: empty"]


def test_same_thread_name_for_two_users_is_two_conversations(make_app):
    app = make_app()
    app.send("demo-user-a", "shared-name", "My issue is Wi-Fi. I enabled it and tried reconnecting.")
    result = app.send("demo-user-b", "shared-name", "Which problem and steps did we discuss?")
    assert "No previous problem" in result["reply"]
    assert app.conversation_state("demo-user-b", "shared-name")["messages"][0].content.startswith("Which problem")


def test_two_threads_of_same_user_keep_separate_histories(make_app):
    app = make_app()
    app.send("demo-user-a", "t1", "My issue is Wi-Fi. I enabled it and tried reconnecting.")
    app.send("demo-user-a", "t2", "I forgot my password.")
    first = app.conversation_state("demo-user-a", "t1")["messages"]
    second = app.conversation_state("demo-user-a", "t2")["messages"]
    assert any("Wi-Fi" in m.content for m in first) and not any("Wi-Fi" in m.content for m in second)


def test_preference_write_is_scoped_to_the_user_in_the_database(storage):
    memory = UserMemory(storage / "user_memory.sqlite")
    memory.set("demo-user-a", "language", "Kazakh")
    assert memory.get_all("demo-user-a") == {"language": "Kazakh"}
    assert memory.get_all("demo-user-b") == {}
