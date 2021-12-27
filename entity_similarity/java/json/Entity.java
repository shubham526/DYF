package json;

import org.jetbrains.annotations.NotNull;
import org.json.JSONException;
import org.json.JSONObject;

public class Entity {
    private final String entityName;
    private final String entityId;
    private final String mention;
    private final boolean targetMention;
    private final long start;
    private final long end;
    public Entity(@NotNull JSONObject jsonObject) throws JSONException {
        this.entityId = jsonObject.getString("entity_id");
        this.entityName = jsonObject.getString("entity_name");
        this.mention = jsonObject.getString("mention");
        this.targetMention = jsonObject.has("target_mention") && jsonObject.getBoolean("target_mention");
        this.start = jsonObject.getLong("start");
        this.end = jsonObject.getLong("end");
    }

    @Override
    public String toString() {
        return "Entity{" +
                "entityName='" + entityName + '\'' +
                ", entityId='" + entityId + '\'' +
                ", mention='" + mention + '\'' +
                ", targetMention=" + targetMention +
                ", start=" + start +
                ", end=" + end +
                '}';
    }

    public String getEntityName() {
        return entityName;
    }

    public String getEntityId() {
        return entityId;
    }

    public String getMention() {
        return mention;
    }

    public boolean isTargetMention() {
        return targetMention;
    }

    public long getStart() {
        return start;
    }

    public long getEnd() {
        return end;
    }
}
