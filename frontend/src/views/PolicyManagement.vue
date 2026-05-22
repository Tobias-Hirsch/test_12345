<template>
  <div class="policy-management-container">
    <h2>ABAC-Richtlinienverwaltung</h2>

    <!-- Richtlinienliste -->
    <el-table :data="policies" style="width: 100%" border>
      <el-table-column prop="id" label="ID" width="80"></el-table-column>
      <el-table-column prop="name" label="Richtlinienname"></el-table-column>
      <el-table-column prop="description" label="Beschreibung"></el-table-column>
      <el-table-column label="Aktiv" width="100">
        <template #default="scope">
          <el-switch
            v-model="scope.row.is_active"
            @change="togglePolicyStatus(scope.row)"
          ></el-switch>
        </template>
      </el-table-column>
      <el-table-column label="Aktionen" width="280">
        <template #default="scope">
          <el-button size="small" @click="editPolicy(scope.row)">Bearbeiten</el-button>
          <el-button size="small" type="primary" plain @click="clonePolicy(scope.row)">Duplizieren</el-button>
          <el-button size="small" type="danger" @click="deletePolicy(scope.row.id)">Löschen</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- Dialog zum Erstellen/Bearbeiten einer Richtlinie -->
    <!-- Dialog zum Erstellen/Bearbeiten einer Richtlinie -->
    <el-dialog
      v-model="dialogVisible"
      :title="dialogTitle"
      width="80%"
      top="5vh"
      destroy-on-close
      @close="handleDialogClose"
    >
      <PolicyBuilder
        v-if="dialogVisible"
        :policy-id="editingPolicyId"
        :initial-data="clonedPolicyData"
        @save="handleSavePolicy"
        @cancel="dialogVisible = false"
      />
    </el-dialog>

    <el-button type="primary" @click="createNewPolicy">Neue Richtlinie erstellen</el-button>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { get, post, put, del } from '@/services/apiService';
import PolicyBuilder from './PolicyManagement/PolicyBuilder.vue';

interface Policy {
  id?: number;
  name: string;
  description: string;
  effect: 'allow' | 'deny';
  actions: string[];
  subjects: any[];
  resources: any[];
  conditions?: any[];
  is_active: boolean;
}

const policies = ref<Policy[]>([]);
const dialogVisible = ref(false);
const isEditMode = ref(false);
const editingPolicyId = ref<number | null>(null);
const clonedPolicyData = ref<Policy | null>(null);

const fetchPolicies = async () => {
  try {
    const data = await get('/policies/');
    if (Array.isArray(data)) {
      policies.value = data.map((policy: Policy) => ({
        ...policy,
        is_active: !!policy.is_active
      }));
    } else {
      console.error('API Fehler bei der Verarbeitung', data);
      policies.value = [];
      ElMessage.error('Fehler bei der Verarbeitung');
    }
  } catch (error: any) {
    console.error('Fehler bei der Verarbeitung', error);
    ElMessage.error(`Fehler beim Laden der Richtlinien: ${error.message || 'Unbekannter Fehler'}`);
  }
};

const createNewPolicy = () => {
  isEditMode.value = false;
  editingPolicyId.value = null;
  clonedPolicyData.value = null;
  dialogVisible.value = true;
};

const editPolicy = (policy: Policy) => {
  isEditMode.value = true;
  editingPolicyId.value = policy.id!;
  clonedPolicyData.value = null;
  dialogVisible.value = true;
};

const clonePolicy = async (policy: Policy) => {
  try {
    const policyToClone = await get(`/policies/${policy.id}`);
    isEditMode.value = false;
    editingPolicyId.value = null;
    clonedPolicyData.value = {
      ...policyToClone,
      id: undefined,
      name: `${policyToClone.name} - Kopie`,
      is_active: false,
    };
    dialogVisible.value = true;
  } catch (error: any) {
    ElMessage.error(`Fehler beim Duplizieren der Richtlinie: ${error.message}`);
  }
};

const handleSavePolicy = async (policyToSave: Omit<Policy, 'id'>) => {
  try {
    // Create a payload with the correct data type for is_active
    const payload = {
      ...policyToSave,
      is_active: policyToSave.is_active ? 1 : 0,
    };

    // Log the final payload before sending to the backend
    console.log('Final payload being sent to API:', JSON.stringify(payload, null, 2));

    if (isEditMode.value && editingPolicyId.value) {
      await put(`/policies/${editingPolicyId.value}`, payload);
      ElMessage.success('Richtlinie gespeichert');
    } else {
      await post('/policies/', payload);
      ElMessage.success('Richtlinie gespeichert');
    }
    dialogVisible.value = false;
    fetchPolicies();
  } catch (error: any) {
    ElMessage.error(`Fehler beim Speichern der Richtlinie: ${error.message}`);
  }
};

const handleDialogClose = () => {
  clonedPolicyData.value = null;
};

const dialogTitle = computed(() => {
  if (isEditMode.value) {
    return 'Richtlinie bearbeiten';
  }
  if (clonedPolicyData.value) {
    return 'Richtlinie duplizieren';
  }
  return 'Neue Richtlinie erstellen';
});

const deletePolicy = async (id: number) => {
  ElMessageBox.confirm('Möchten Sie diese Richtlinie wirklich löschen?', 'Warnung', {
    confirmButtonText: 'OK',
    cancelButtonText: 'Abbrechen',
    type: 'warning',
  })
    .then(async () => {
      try {
        await del(`/policies/${id}`);
        ElMessage.success('Richtlinie gelöscht');
        fetchPolicies();
      } catch (error: any) {
        ElMessage.error(`Fehler beim Löschen der Richtlinie: ${error.message}`);
      }
    })
    .catch(() => {
      ElMessage.info('Löschen abgebrochen');
    });
};

const togglePolicyStatus = async (policy: Policy) => {
  try {
    await put(`/policies/${policy.id}`, { is_active: policy.is_active });
    ElMessage.success('Richtlinie gespeichert');
  } catch (error: any) {
    ElMessage.error(`Fehler beim Aktualisieren des Richtlinienstatus: ${error.message}`);
    policy.is_active = !policy.is_active;
  }
};

// These functions are no longer needed as we have separate fields
// const formatJson = ...
// const validateJson = ...

onMounted(() => {
  fetchPolicies();
});
</script>