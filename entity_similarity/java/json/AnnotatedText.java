package json;

import org.jetbrains.annotations.NotNull;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class AnnotatedText {
    private final String content;
    private final List<Entity> entities = new ArrayList<>();

    public AnnotatedText(@NotNull JSONObject jsonObject) throws JSONException {
        this.content = jsonObject.getString("content");
        JSONArray entityList = jsonObject.getJSONArray("entities");

        for (int i = 0; i < entityList.length(); i++) {
            this.entities.add(new Entity(entityList.getJSONObject(i)));
        }
    }

    @Override
    public String toString() {
        return "AnnotatedText{" +
                "content='" + content + '\'' +
                ", entities=" + entities +
                '}';
    }

    public String getContent() {
        return content;
    }

    public List<Entity> getEntities() {
        return entities;
    }
}
