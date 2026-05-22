<template>
  <div class="policy-builder-container">
    <el-form :model="policy" label-position="top">
      <h3>Basisinformationen</h3>
      <el-row :gutter="20">
        <el-col :span="12">
          <el-form-item label="Richtlinienname">
            <el-input v-model="policy.name" placeholder="Eingabe"></el-input>
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="Wirkung">
            <el-select v-model="policy.effect" style="width: 100%;">
              <el-option label="Erlauben (Allow)" value="allow"></el-option>
              <el-option label="Ablehnen (Deny)" value="deny"></el-option>
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>
      <el-form-item label="Beschreibung">
        <el-input type="textarea" v-model="policy.description" placeholder="Eingabe"></el-input>
      </el-form-item>

      <el-divider></el-divider>

      <h3>Regeldefinition: Richtliniensatz erstellen</h3>
      <p class="policy-sentence">
        Diese Richtlinie <strong>{{ policy.effect === 'allow' ? 'erlaubt' : 'verweigert' }}</strong>
        <strong>[Subjekt]</strong> den Zugriff auf <strong>[Ressource]</strong>
        für <strong>[Aktionen]</strong>, wenn die definierten Bedingungen erfüllt sind.
      </p>

      <!-- Subjekte (Subjects) -->
      <div class="rule-block">
        <h4>Subjekt (Subject)</h4>
        <p class="description">Definiert, wer von dieser Richtlinie betroffen ist. Ein Subjekt muss alle angegebenen Regeln erfüllen.</p>
        
        <div v-for="(rule, index) in subjectRules" :key="index" class="rule-row">
          <el-select v-model="rule.key" placeholder="Subjekteigenschaft auswählen" filterable style="width: 250px;" @change="onAttributeChange(rule, 'subject')">
            <el-option
              v-for="attr in subjectAttributes"
              :key="attr.key"
              :label="attr.name"
              :value="attr.key">
              <span style="float: left">{{ attr.name }}</span>
              <span style="float: right; color: #8492a6; font-size: 13px">{{ attr.key }}</span>
            </el-option>
          </el-select>

          <el-select v-model="rule.operator" placeholder="Operator auswählen" style="width: 150px; margin-left: 10px;">
            <el-option
              v-for="op in getOperatorsForType(rule.type)"
              :key="op.value"
              :label="op.label"
              :value="op.value">
            </el-option>
          </el-select>

          <el-select
            v-if="isValueASelection(rule.key)"
            v-model="rule.value"
            filterable
            allow-create
            default-first-option
            placeholder="Wert auswählen oder eingeben"
            style="width: 300px; margin-left: 10px;"
          >
            <el-option
              v-for="item in getValueOptions(rule.key)"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-input v-else v-model="rule.value" placeholder="Erwarteten Wert eingeben" style="width: 300px; margin-left: 10px;" />

          <el-button type="danger" :icon="ElIconDelete" circle plain @click="removeSubjectRule(index)" style="margin-left: 10px;" />
        </div>

        <el-button @click="addSubjectRule" :icon="ElIconPlus">Subjektregel hinzufügen</el-button>
      </div>

      <!-- Ressource (Resources) -->
      <div class="rule-block">
        <h4>Ressource (Resource)</h4>
        <p class="description">Definiert, worauf diese Richtlinie angewendet wird. Eine Ressource muss alle angegebenen Regeln erfüllen.</p>

        <div v-for="(rule, index) in resourceRules" :key="index" class="rule-row">
          <el-select v-model="rule.key" placeholder="Eingabe" filterable style="width: 250px;" @change="onAttributeChange(rule, 'resource')">
            <el-option
              v-for="attr in resourceAttributes"
              :key="attr.key"
              :label="attr.name"
              :value="attr.key">
              <span style="float: left">{{ attr.name }}</span>
              <span style="float: right; color: #8492a6; font-size: 13px">{{ attr.key }}</span>
            </el-option>
          </el-select>

          <el-select v-model="rule.operator" placeholder="Operator auswählen" style="width: 150px; margin-left: 10px;">
            <el-option
              v-for="op in getOperatorsForType(rule.type)"
              :key="op.value"
              :label="op.label"
              :value="op.value">
            </el-option>
          </el-select>

          <el-select
            v-if="isValueASelection(rule.key)"
            v-model="rule.value"
            filterable
            allow-create
            default-first-option
            placeholder="Wert auswählen oder eingeben"
            style="width: 300px; margin-left: 10px;"
          >
            <el-option
              v-for="item in getValueOptions(rule.key)"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-input v-else v-model="rule.value" placeholder="Erwarteten Wert eingeben" style="width: 300px; margin-left: 10px;" />

          <el-button type="danger" :icon="ElIconDelete" circle plain @click="removeResourceRule(index)" style="margin-left: 10px;" />
        </div>

        <el-button @click="addResourceRule" :icon="ElIconPlus">Ressourcenregel hinzufügen</el-button>
      </div>

      <!-- Aktionen (Actions) -->
      <div class="rule-block">
        <h4>Aktionen (Action)</h4>
        <p class="description">Definiert die konkret erlaubte oder verweigerte Aktion.</p>
        <el-checkbox-group v-model="policy.actions">
          <el-checkbox v-for="action in availableActions" :key="action" :label="action" :value="action">
            {{ action }}
          </el-checkbox>
        </el-checkbox-group>
      </div>

      <!-- Bedingung (Conditions) - Optional -->
      <div class="rule-block">
        <h4>Bedingung (Condition) <el-tag size="small">Optional</el-tag></h4>
        <p class="description">Definiert zusätzliche Einschränkungen, die für die Gültigkeit der Richtlinie erfüllt sein müssen. Bedingungen sind optional.</p>
        
        <div v-for="(rule, index) in conditionRules" :key="index" class="rule-row">
          <el-select v-model="rule.key" placeholder="Eingabe" filterable style="width: 250px;" @change="onAttributeChange(rule, 'condition')">
            <el-option
              v-for="attr in conditionAttributes"
              :key="attr.key"
              :label="attr.name"
              :value="attr.key">
              <span style="float: left">{{ attr.name }}</span>
              <span style="float: right; color: #8492a6; font-size: 13px">{{ attr.key }}</span>
            </el-option>
          </el-select>

          <el-select v-model="rule.operator" placeholder="Operator auswählen" style="width: 150px; margin-left: 10px;">
            <el-option
              v-for="op in getOperatorsForType(rule.type)"
              :key="op.value"
              :label="op.label"
              :value="op.value">
            </el-option>
          </el-select>

          <el-select
            v-if="isValueASelection(rule.key)"
            v-model="rule.value"
            filterable
            allow-create
            default-first-option
            placeholder="Wert auswählen oder eingeben"
            style="width: 300px; margin-left: 10px;"
          >
            <el-option
              v-for="item in getValueOptions(rule.key)"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-input v-else v-model="rule.value" placeholder="Erwarteten Wert eingeben" style="width: 300px; margin-left: 10px;" />

          <el-button type="danger" :icon="ElIconDelete" circle plain @click="removeConditionRule(index)" style="margin-left: 10px;" />
        </div>

        <el-button @click="addConditionRule" :icon="ElIconPlus">Bedingung hinzufügen</el-button>
      </div>

      <el-divider />

      <div class="form-footer">
        <el-button @click="cancel">Abbrechen</el-button>
        <el-button type="primary" @click="savePolicy">Speichern</el-button>
      </div>

    </el-form>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, computed, PropType } from 'vue';
import { get, getRoles } from '@/services/apiService';
import { ElMessage } from 'element-plus';
import { Delete as ElIconDelete, Plus as ElIconPlus } from '@element-plus/icons-vue';

// Interface for a single rule in the builder
interface Rule {
  key: string;
  operator: string;
  value: string;
  type: string; // The data type of the attribute, e.g., 'string', 'integer'
}

// Interface for the policy object being built
interface PolicyBuilderState {
  id?: number;
  name: string;
  description: string;
  effect: 'allow' | 'deny';
  actions: string[];
  subjects: any[]; // To be defined with more specific types
  resources: any[]; // To be defined with more specific types
  conditions?: any[]; // To be defined with more specific types
  is_active: boolean;
}

// Props for the component, e.g., to load an existing policy for editing
const props = defineProps({
  policyId: {
    type: Number as PropType<number | null>,
    default: null,
  },
  initialData: {
    type: Object as PropType<PolicyBuilderState | null>,
    default: null,
  },
});

const emit = defineEmits(['save', 'cancel']);

const policy = ref<PolicyBuilderState>({
  name: '',
  description: '',
  effect: 'allow',
  actions: [],
  subjects: [],
  resources: [],
  conditions: [],
  is_active: true,
});

// --- Start of Subject Rules Logic ---
const subjectRules = ref<Rule[]>([]);

const subjectAttributes = computed(() =>
  availableAttributes.value.filter((attr: any) => attr.category === 'subject')
);

const addSubjectRule = () => {
  subjectRules.value.push({ key: '', operator: 'equals', value: '', type: 'string' });
};

const removeSubjectRule = (index: number) => {
  subjectRules.value.splice(index, 1);
};

watch(subjectRules, (newRules) => {
  policy.value.subjects = newRules
    .filter(rule => rule.key && rule.value)
    .map(rule => ({
      key: rule.key,
      operator: rule.operator,
      value: [rule.value] // Backend expects a list of strings
    }));
}, { deep: true });
// --- End of Subject Rules Logic ---

// --- Start of Resource Rules Logic ---
const resourceRules = ref<Rule[]>([]);

const resourceAttributes = computed(() =>
  availableAttributes.value.filter((attr: any) => attr.category === 'resource')
);

const addResourceRule = () => {
  resourceRules.value.push({ key: '', operator: 'equals', value: '', type: 'string' });
};

const removeResourceRule = (index: number) => {
  resourceRules.value.splice(index, 1);
};

watch(resourceRules, (newRules) => {
  policy.value.resources = newRules
    .filter(rule => rule.key && rule.value)
    .map(rule => ({
      key: rule.key,
      operator: rule.operator,
      value: [rule.value] // Backend expects a list of strings
    }));
}, { deep: true });
// --- End of Resource Rules Logic ---

// --- Start of Condition Rules Logic ---
const conditionRules = ref<Rule[]>([]);

// Conditions can use any attribute
const conditionAttributes = computed(() => availableAttributes.value);

const addConditionRule = () => {
  conditionRules.value.push({ key: '', operator: 'equals', value: '', type: 'string' });
};

const removeConditionRule = (index: number) => {
  conditionRules.value.splice(index, 1);
};

watch(conditionRules, (newRules) => {
  if (newRules.length === 0 || newRules.every(rule => !rule.key || !rule.value)) {
    policy.value.conditions = [];
    return;
  }
  policy.value.conditions = newRules
    .filter(rule => rule.key && rule.value)
    .map(rule => ({
      key: rule.key, // 'key' is the correct field name per schema
      operator: rule.operator,
      value: [rule.value] // Backend expects a list of strings
    }));
}, { deep: true });
// --- End of Condition Rules Logic ---


// Data stores for the "vocabulary" from the backend
const availableAttributes = ref<any[]>([]);
const availableActions = ref([]);
const availableResourceTypes = ref([]);
const availableRoles = ref<{ name: string }[]>([]);

// Fetching the vocabulary from our new backend APIs
const fetchVocabulary = async () => {
  try {
    const [attributes, actions, resourceTypes, roles] = await Promise.all([
      get('/abac/attributes'),
      get('/abac/actions'),
      get('/abac/resource-types'),
      getRoles(),
    ]);
    availableAttributes.value = attributes;
    availableActions.value = actions;
    availableResourceTypes.value = resourceTypes;
    availableRoles.value = roles;
  } catch (error) {
    console.error('Failed to fetch policy vocabulary:', error);
    ElMessage.error('Fehler bei der Verarbeitung');
  }
};

const setPolicyForEditing = (p: any) => {
  // This function takes a policy object from the API and populates the builder's state.
  policy.value.id = p.id;
  policy.value.name = p.name;
  policy.value.description = p.description;
  policy.value.effect = p.effect;
  policy.value.actions = p.actions;
  policy.value.is_active = !!p.is_active;

  // Reverse-transform subjects from API format to UI format (Rule[])
  subjectRules.value = p.subjects.map((filter: any) => {
    const attribute = availableAttributes.value.find(attr => attr.key === filter.key);
    return {
      key: filter.key,
      operator: filter.operator,
      value: filter.value[0] || '', // Take the first value
      type: attribute ? attribute.type : 'string',
    };
  });

  // Reverse-transform resources
  resourceRules.value = p.resources.map((filter: any) => {
    const attribute = availableAttributes.value.find(attr => attr.key === filter.key);
    return {
      key: filter.key,
      operator: filter.operator,
      value: filter.value[0] || '',
      type: attribute ? attribute.type : 'string',
    };
  });

  // Reverse-transform conditions
  // The API response uses 'conditions', which is correct.
  conditionRules.value = (p.conditions || []).map((filter: any) => {
    const attribute = availableAttributes.value.find(attr => attr.key === filter.key);
    return {
      key: filter.key,
      operator: filter.operator,
      value: filter.value[0] || '',
      type: attribute ? attribute.type : 'string',
    };
  });
};

onMounted(async () => {
  // Always fetch vocabulary first
  await fetchVocabulary();

  if (props.initialData) {
    // 1. Priority: Use initial data if provided (for cloning)
    setPolicyForEditing(props.initialData);
  } else if (props.policyId) {
    // 2. Fallback: Fetch policy by ID if provided (for editing)
    try {
      const existingPolicy = await get(`/policies/${props.policyId}`);
      if (existingPolicy) {
        setPolicyForEditing(existingPolicy);
      }
    } catch (error) {
      ElMessage.error('Fehler bei der Verarbeitung');
      console.error(`Failed to fetch policy ${props.policyId} for editing:`, error);
    }
  }
  // 3. If neither, it's a new policy, and vocabulary is already fetched.
});

const savePolicy = () => {
  // Basic validation
  if (!policy.value.name) {
    ElMessage.warning('RichtliniennameWarnhinweis');
    return;
  }
  if (policy.value.actions.length === 0) {
    ElMessage.warning('Warnhinweis');
    return;
  }

  // The policy ref already contains the structured data.
  // We just need to emit it.
  emit('save', { ...policy.value });
};

const cancel = () => {
  emit('cancel');
};

// --- Helper functions for dynamic UI ---
const onAttributeChange = (rule: Rule, category: 'subject' | 'resource' | 'condition') => {
  const selectedAttr = availableAttributes.value.find(attr => attr.key === rule.key);
  if (selectedAttr) {
    rule.type = selectedAttr.type;
    // Reset operator and value
    rule.operator = getOperatorsForType(rule.type)[0].value;
    rule.value = '';
  }
};

const getOperatorsForType = (type: string) => {
  const commonOperators = [
    { label: 'Beschriftung', value: 'eq' },
    { label: 'Beschriftung', value: 'not_eq' }, // Assuming backend supports 'not_eq'
  ];
  const arrayOperators = [
    { label: 'Beschriftung', value: 'in' },
    { label: 'Beschriftung', value: 'not_in' }, // Assuming backend supports 'not_in'
  ];
  const numericOperators = [
    { label: 'Beschriftung', value: 'greater_than' },
    { label: 'Beschriftung', value: 'less_than' },
  ];

  switch (type) {
    case 'array_string':
    case 'array_integer':
      return [...commonOperators, ...arrayOperators];
    case 'integer':
      return [...commonOperators, ...numericOperators];
    default: // string, etc.
      return commonOperators;
  }
};
// --- End of Helper functions ---

const isValueASelection = (key: string) => {
  return key === 'user.roles' || key === 'resource.type';
};

const getValueOptions = (key: string) => {
  if (key === 'user.roles') {
    return availableRoles.value.map(role => ({ label: role.name, value: role.name }));
  }
  if (key === 'resource.type') {
    return availableResourceTypes.value.map((rt: string) => ({ label: rt, value: rt }));
  }
  return [];
};

</script>

<style scoped>
.policy-builder-container {
  padding: 20px;
}
.form-footer {
  display: flex;
  justify-content: flex-end;
  margin-top: 20px;
}
.rule-block {
  border: 1px solid #ebeef5;
  border-radius: 4px;
  padding: 20px;
  margin-bottom: 20px;
  background-color: #fafafa;
}
.rule-row {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}
.policy-sentence {
  font-size: 1.1em;
  color: #606266;
  margin-bottom: 20px;
  padding: 10px;
  background-color: #f0f9eb;
  border-left: 5px solid #67c23a;
}
.description {
  font-size: 0.9em;
  color: #909399;
  margin-top: 0;
}
h3, h4 {
    margin-bottom: 10px;
}
</style>